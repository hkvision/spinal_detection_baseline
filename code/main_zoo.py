import time
import argparse

from zoo import init_spark_on_local, init_spark_on_yarn
from zoo.ray import RayContext
from zoo.orca.learn.pytorch import PyTorchTrainer, Estimator
from zoo.orca.learn.pytorch.training_operator import TrainingOperator


class EvaluateOperator(TrainingOperator):
    # This is to make the TrainingOperator works for both PyTorchTrainer and Estimator.
    def __init__(self,
                 config,
                 models,
                 optimizers,
                 world_rank,
                 train_loader=None,
                 validation_loader=None,
                 criterion=None,
                 schedulers=None,
                 device_ids=None,
                 use_gpu=False,
                 use_fp16=False,
                 use_tqdm=False):
        super(EvaluateOperator, self).__init__(config, models, optimizers, world_rank, criterion,
                                               schedulers, device_ids, use_gpu, use_fp16, use_tqdm)

    def validate(self, val_iterator, info):
        import sys
        sys.path.append('spinal_detection_baseline')
        from code.core.disease.evaluation import Evaluator
        from ray.util.sgd.utils import AverageMeterCollection

        metric_meters = AverageMeterCollection()
        valid_evaluator = Evaluator(
            self.model, val_iterator, 'data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
        )
        valid_data = None
        metrics_values = valid_evaluator(self.model, valid_data, [valid_evaluator.metric])
        metric_value = metrics_values[0][1]
        print(metric_value)
        metrics_dict = {}
        for result in metrics_values:
            metrics_dict[result[0]] = result[1]
        metric_meters.update(metrics_dict, n=len(val_iterator)*20)  # 20 is num_rep
        return metric_meters.summary()


def train_data_creator(config):
    import sys
    sys.path.append('spinal_detection_baseline')
    from code.core.disease.data_loader import DisDataLoader
    from code.core.structure import construct_studies

    train_studies, train_annotation, train_counter = construct_studies(
        'data/lumbar_train150', 'data/lumbar_train150_annotation.json', multiprocessing=True)
    train_dataloader = DisDataLoader(
        train_studies, train_annotation, batch_size=config["batch_size"], num_workers=3,
        num_rep=1, prob_rotate=1, max_angel=180, sagittal_size=config["sagittal_size"],
        transverse_size=config["sagittal_size"], k_nearest=0)
    return train_dataloader


def val_data_creator(config):
    import sys
    sys.path.append('spinal_detection_baseline')
    from code.core.structure import construct_studies
    valid_studies, valid_annotation, valid_counter = construct_studies(
        'data/train/', 'data/lumbar_train51_annotation.json', multiprocessing=True)
    return valid_studies


def data_creator(config):
    return train_data_creator(config), val_data_creator(config)


def model_creator(config):
    import sys
    sys.path.append('spinal_detection_baseline')
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    from code.core.key_point import KeyPointModel
    from code.core.disease.model import DiseaseModelBase

    backbone = resnet_fpn_backbone('resnet50', True)
    kp_model = KeyPointModel(backbone)
    dis_model = DiseaseModelBase(kp_model, sagittal_size=config["sagittal_size"])
    print(dis_model)
    return dis_model


def optimizer_creator(model, config):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=1e-5)

def loss_creator(config):
    import sys
    sys.path.append('spinal_detection_baseline')
    # Since NullLoss is not a TorchLoss, it can't be directly passed into PyTorchTrainer.
    from code.core.key_point import NullLoss
    return NullLoss()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default="horovod",
                        help='The backend to use for distributed training.')
    parser.add_argument('--num_executors', '-n', type=int, default=1,
                        help="Sets number of workers for training.")
    parser.add_argument('--executor_cores', type=int, default=4,
                        help='The number of executor cores you want to use.')
    parser.add_argument('--hadoop_conf', type=str,
                        help='The path to the hadoop configuration folder. Required if you '
                             'wish to run on yarn clusters. Otherwise, run in local mode.')
    parser.add_argument('--conda_name', type=str,
                        help='The name of conda environment. Required if you '
                             'wish to run on yarn clusters.')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='The number of epochs to train the model.')

    opt, _ = parser.parse_known_args()

    if opt.hadoop_conf:
        assert opt.conda_name is not None, "conda_name must be specified for yarn mode"
        sc = init_spark_on_yarn(
            hadoop_conf=opt.hadoop_conf,
            conda_name=opt.conda_name,
            num_executors=opt.num_executors,
            executor_cores=opt.executor_cores,
            additional_archive="data.zip#data,resnet.zip#resnet,spinal_detection_baseline.zip#spinal_detection_baseline")
        ray_ctx = RayContext(sc=sc,
                             env={"RAY_BACKEND_LOG_LEVEL": "debug",
                                  # num_cpus for each runner is hardcoded to be 1 in RaySGD TorchTrainer.
                                  "OMP_NUM_THREADS": str(opt.executor_cores),
                                  "KMP_AFFINITY": "granularity=fine,compact,1,0",
                                  # Downloading the resnet model on every node is slow and would sometimes
                                  # have connection error. Upload the model in archive so that each worker
                                  # can directly load the checkpoint.
                                  "TORCH_HOME": "resnet/"})
    else:
        sc = init_spark_on_local(cores="*")
        ray_ctx = RayContext(sc=sc, env={"RAY_BACKEND_LOG_LEVEL": "debug"})

    ray_ctx.init()

    train_args = {"model_creator": model_creator,
                  "optimizer_creator": optimizer_creator,
                  "loss_creator": loss_creator,
                  "training_operator_cls": EvaluateOperator,
                  "config": {"sagittal_size": (512, 512),
                             "batch_size": 8}}
    start_time = time.time()
    if opt.backend == "horovod":
        estimator = Estimator.from_model_creator(**train_args)
        train_stats = estimator.fit(train_data_creator, epochs=opt.epochs)
        val_stats = estimator.evaluate(val_data_creator)
    else:
        train_args["data_creator"] = data_creator
        train_args["num_workers"] = opt.num_executors
        train_args["use_tqdm"] = True
        trainer = PyTorchTrainer(**train_args)
        train_stats = trainer.train(nb_epoch=opt.epochs)
        val_stats = trainer.validate()

    print("train stats: {}".format(train_stats))
    print("validation stats: {}".format(val_stats))

    # torch.save(dis_model.cpu().state_dict(), 'models/baseline.dis_model')
    # # 预测
    # testA_studies = construct_studies('data/lumbar_testA50/', multiprocessing=True)
    #
    # result = []
    # for study in testA_studies.values():
    #     result.append(dis_model.eval()(study, True))
    #
    # with open('predictions/baseline.json', 'w') as file:
    #     json.dump(result, file)
    print('task completed, {} seconds used'.format(time.time() - start_time))
