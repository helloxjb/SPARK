import numpy as np

import collections
import logging
from data_1.datasets import (
    APY, CUB200, LAD, AwA2, ImageNet100, ImageNet100UCIR, ImageNet1000, TinyImageNet200, iCIFAR10,
    iCIFAR100
)
from lib import factory
from torchvision import datasets, transforms
import warnings

# 忽略特定类型的警告
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()` was not necessary and is being deprecated where possible.", category=UserWarning)


logger = logging.getLogger(__name__)







class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = [transforms.ToTensor()]
    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63 / 255)
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)


class iCIFAR100(iCIFAR10):
    base_dataset = datasets.cifar.CIFAR100
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    class_order = [  # Taken from original iCaRL implementation:
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]


class iMNIST(DataHandler):
    base_dataset = datasets.MNIST
    train_transforms = [transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()]
    common_transforms = [transforms.ToTensor()]

def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet100":
        return ImageNet100
    elif dataset_name == "imagenet100ucir":
        return ImageNet100UCIR
    elif dataset_name == "imagenet1000":
        return ImageNet1000
    elif dataset_name == "tinyimagenet":
        return TinyImageNet200
    elif dataset_name == "awa2":
        return AwA2
    elif dataset_name == "cub200":
        return CUB200
    elif dataset_name == "apy":
        return APY
    elif dataset_name == "lad":
        return LAD
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

import parser_1
import network
import util



def _set_global_parameters(config):
    _set_seed(config["seed"], config["threads"], config["no_benchmark"], config["detect_anomaly"])
    factory.set_device(config)

def _set_data_model(config, class_order):
    inc_dataset = factory.get_data(config, class_order)
    config["classes_order"] = inc_dataset.class_order

    model = factory.get_model(config)
    model.inc_dataset = inc_dataset

    return inc_dataset, model

def train(args, class_order):
    _set_global_parameters(args)
    inc_dataset, model = _set_data_model(args, class_order)
    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )
    class_mean = []
    for task_id in range(inc_dataset.n_tasks):
        if task_id == 0:
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
            print(task_info)

            model.set_task_info(task_info)

            # ---------------
            # 1. Prepare Task
            # ---------------
            model.eval()
            model.before_task(train_loader, val_loader if val_loader else test_loader)

            # -------------
            # 2. Train Task
            # -------------
            _train_task(args, model, train_loader, val_loader, test_loader, results_folder, run_id, task_info, task_id,
                        class_mean)

            # ----------------
            # 3. Conclude Task
            # ----------------
            model.eval()
            _after_task(args, model, inc_dataset, run_id, task_id, results_folder)
            class_mean = model._class_means

            # ------------
            # 4. Eval Task
            # ------------
            logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
            ypreds, ytrue = model.eval_task(test_loader)
            metric_logger.log_task(
                ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
            )

            if args["dump_predictions"] and args["label"]:
                os.makedirs(
                    os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
                )
                with open(
                        os.path.join(
                            results_folder, "predictions_{}".format(run_id),
                            str(task_id).rjust(len(str(30)), "0") + ".pkl"
                        ), "wb+"
                ) as f:
                    pickle.dump((ypreds, ytrue), f)

            if args["label"]:
                logger.info(args["label"])
            logger.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
            logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
            logger.info(
                "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
            )
            logger.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
            logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
            logger.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
            if task_id > 0:
                logger.info(
                    "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                        metric_logger.last_results["old_accuracy"],
                        metric_logger.last_results["avg_old_accuracy"]
                    )
                )
                logger.info(
                    "New accuracy: {:.2f}, mean: {:.2f}.".format(
                        metric_logger.last_results["new_accuracy"],
                        metric_logger.last_results["avg_new_accuracy"]
                    )
                )
            if args.get("all_test_classes"):
                logger.info(
                    "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
                )
                logger.info(
                    "unSeen classes: {:.2f}.".format(
                        metric_logger.last_results["unseen_classes_accuracy"]
                    )
                )

            results["results"].append(metric_logger.last_results)

            avg_inc_acc = results["results"][-1]["incremental_accuracy"]
            last_acc = results["results"][-1]["accuracy"]["total"]
            forgetting = results["results"][-1]["forgetting"]
            yield avg_inc_acc, last_acc, forgetting

            memory = model.get_memory()
            memory_val = model.get_val_memory()
        else:
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)

            model.set_task_info(task_info)

            # ---------------
            # 1. Prepare Task
            # ---------------
            model.eval()
            model.before_task(train_loader, val_loader if val_loader else test_loader)

            # -------------
            # 2. Train Task
            # -------------
            _train_task(args, model, train_loader, val_loader, test_loader, results_folder, run_id, task_info, task_id,
                        class_mean)

            # ----------------
            # 3. Conclude Task
            # ----------------
            model.eval()
            _after_task(args, model, inc_dataset, run_id, task_id, results_folder)

            # ------------
            # 4. Eval Task
            # ------------
            logger.info("Eval on {}->{}.".format(0, task_info["max_class"]))
            ypreds, ytrue = model.eval_task(test_loader)
            metric_logger.log_task(
                ypreds, ytrue, task_size=task_info["increment"], zeroshot=args.get("all_test_classes")
            )

            if args["dump_predictions"] and args["label"]:
                os.makedirs(
                    os.path.join(results_folder, "predictions_{}".format(run_id)), exist_ok=True
                )
                with open(
                        os.path.join(
                            results_folder, "predictions_{}".format(run_id),
                            str(task_id).rjust(len(str(30)), "0") + ".pkl"
                        ), "wb+"
                ) as f:
                    pickle.dump((ypreds, ytrue), f)

            if args["label"]:
                logger.info(args["label"])
            logger.info("Avg inc acc: {}.".format(metric_logger.last_results["incremental_accuracy"]))
            logger.info("Current acc: {}.".format(metric_logger.last_results["accuracy"]))
            logger.info(
                "Avg inc acc top5: {}.".format(metric_logger.last_results["incremental_accuracy_top5"])
            )
            logger.info("Current acc top5: {}.".format(metric_logger.last_results["accuracy_top5"]))
            logger.info("Forgetting: {}.".format(metric_logger.last_results["forgetting"]))
            logger.info("Cord metric: {:.2f}.".format(metric_logger.last_results["cord"]))
            if task_id > 0:
                logger.info(
                    "Old accuracy: {:.2f}, mean: {:.2f}.".format(
                        metric_logger.last_results["old_accuracy"],
                        metric_logger.last_results["avg_old_accuracy"]
                    )
                )
                logger.info(
                    "New accuracy: {:.2f}, mean: {:.2f}.".format(
                        metric_logger.last_results["new_accuracy"],
                        metric_logger.last_results["avg_new_accuracy"]
                    )
                )
            if args.get("all_test_classes"):
                logger.info(
                    "Seen classes: {:.2f}.".format(metric_logger.last_results["seen_classes_accuracy"])
                )
                logger.info(
                    "unSeen classes: {:.2f}.".format(
                        metric_logger.last_results["unseen_classes_accuracy"]
                    )
                )

            results["results"].append(metric_logger.last_results)

            avg_inc_acc = results["results"][-1]["incremental_accuracy"]
            last_acc = results["results"][-1]["accuracy"]["total"]
            forgetting = results["results"][-1]["forgetting"]
            yield avg_inc_acc, last_acc, forgetting

            memory = model.get_memory()
            memory_val = model.get_val_memory()

    # for task_id in range(inc_dataset.n_tasks):

    logger.info(
        "Average Incremental Accuracy: {}.".format(results["results"][-1]["incremental_accuracy"])
    )
    if args["label"] is not None:
        results_utils.save_results(
            results, args["label"], args["model"], start_date, run_id, args["seed"]
        )

    del model
    del inc_dataset


# ------------------------
# Lifelong Learning phases
# ------------------------


def _train_task(config, model, train_loader, val_loader, test_loader, results_folder, run_id, task_info, task_id,
                class_mean):
    if config["resume"] is not None and os.path.isdir(config["resume"]) \
            and ((config["resume_first"] and task_id == 0) or not config["resume_first"]):
        model.load_parameters(config["resume"], run_id)
        logger.info(
            "Skipping training phase {} because reloading pretrained model.".format(task_id)
        )
    elif config["resume"] is not None and os.path.isfile(config["resume"]) and \
            os.path.exists(config["resume"]) and task_id == 0:
        # In case we resume from a single model file, it's assumed to be from the first task.
        model.network = config["resume"]
        logger.info(
            "Skipping initial training phase {} because reloading pretrained model.".
                format(task_id)
        )
    else:
        logger.info("Train on {}->{}.".format(task_info["min_class"], task_info["max_class"]))
        model.train()
        # run_id = 1
        # path = os.path.join(f"meta_{run_id}_task_{task_id}.pkl")
        # f = open(path, "rb")
        # class_means = pickle.load(f)[-1]

        model._train_task(train_loader, val_loader if val_loader else test_loader, task_id, class_mean)


def _after_task(config, model, inc_dataset, run_id, task_id, results_folder):
    if config["resume"] and os.path.isdir(config["resume"]) and not config["recompute_meta"] \
            and ((config["resume_first"] and task_id == 0) or not config["resume_first"]):
        model.load_metadata(config["resume"], run_id)
        print(model._class_means)
    else:
        model.after_task_intensive(inc_dataset)

    model.after_task(inc_dataset)

    if config["label"] and (
            config["save_model"] == "task" or
            (config["save_model"] == "last" and task_id == inc_dataset.n_tasks - 1) or
            (config["save_model"] == "first" and task_id == 0)
    ):
        model.save_parameters(results_folder, run_id)
        model.save_metadata(results_folder, run_id)


# ----------
# Parameters
# ----------


def _set_results(config, start_date):
    if config["label"]:
        results_folder = results_utils.get_save_folder(config["model"], start_date, config["label"])
    else:
        results_folder = None

    if config["save_model"]:
        logger.info("Model will be save at this rythm: {}.".format(config["save_model"]))

    results = results_utils.get_template_results(config)

    return results, results_folder





def main_1():
    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    datasets = _get_dataset(args["dataset"])
    train_idx = _get_dataset(args["dataset"]).class_order
    data_path = "data_1"
    inc_dataset = factory.get_data(args, train_idx)
    #train_dataset = datasets().base_dataset.data
    #print(train_dataset)
    train_dataset = datasets().base_dataset(data_path, train=True, download=False)
    test_dataset = datasets().base_dataset(data_path, train=False, download=False)
    x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
    x_val, y_val, x_train, y_train = inc_dataset._split_per_class(
        x_train, y_train, validation_split=0
    )
    x_test, y_test = test_dataset.data, np.array(test_dataset.targets)
    order = _get_dataset(args["dataset"]).class_order
    y_train = inc_dataset._map_new_class_index(y_train, order)
    y_val = inc_dataset._map_new_class_index(y_val, order)
    y_test = inc_dataset._map_new_class_index(y_test, order)
    print(len(y_train), len(y_val), len(y_test))
    model = factory.get_model(args)
    memory, memory_val = None, None
    for task_id in range(inc_dataset.n_tasks):
        if task_id == 0:
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
            print(task_info)
    print(model)

#def knn_gragh_matrx(connect_idx):
    #for i in nnnnnnnnnnnnnnnnnnnnn
def dis2percent(dis_list):
    dis_ = [0.0]
    for i in range(1, len(dis_list)):
        dis_.append(1/dis_list[i])
    return [(dis_[i]/np.sum(dis_)) for i in range(len(dis_))]
def _clean_list(l):
    for i in range(len(l)):
        l[i] = None


from network import hook
from torch import nn
from sklearn.neighbors import NearestNeighbors
from lib import metrics
from tqdm import tqdm
from models.gcn import GCN
def main_4():
    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    train_idx = _get_dataset(args["dataset"]).class_order
    print(train_idx)
    _device = args["device"][0]
    _rotations_config = args.get("rotations_config", {})
    inc_dataset = factory.get_data(args, train_idx)
    # print(inc_dataset)
    _network = network.BasicNet(
        args["convnet"],
        convnet_kwargs=args.get("convnet_config", {}),
        classifier_kwargs=args.get("classifier_config", {
            "type": "fc",
            "use_bias": True
        }),
        device=_device,
        extract_no_act=True,
        classifier_no_act=False,
        rotations_predictor=bool(_rotations_config)
    )
    model = factory.get_model(args)
    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )

    for task_id in range(inc_dataset.n_tasks):
        grad, act = None, None
        clipper = None
        if len(model._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(model._multiple_devices)))
            training_network = nn.DataParallel(model._network, model._multiple_devices)
            if model._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = model._network
        if task_id == 0:
            model._data_memory, model._targets_memory = None, None
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)

        model._metrics = collections.defaultdict(float)

        model.set_task_info(task_info)
        # ---------------
        # 1. Prepare Task
        # ---------------
        model.eval()
        model.before_task(train_loader, val_loader if val_loader else test_loader)

        # -------------
        # 2. Train Task
        # -------------

        prog_bar = tqdm(
            train_loader,
            disable=model._disable_progressbar,
            ascii=True,
            bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
        )
        for i, input_dict in enumerate(prog_bar, start=1):
            inputs, targets = input_dict["inputs"], input_dict["targets"]
            memory_flags = input_dict["memory_flags"]

            if grad is not None:
                _clean_list(grad)
                _clean_list(act)

            model._optimizer.zero_grad()
            loss = model._forward_loss_1(
                training_network,
                inputs,
                targets,
                memory_flags,
                gradcam_grad=grad,
                gradcam_act=act
            )
            loss.backward()
            model._optimizer.step()

            if clipper:
                training_network.apply(clipper)

            #model._print_metrics(prog_bar, i, epoch=1,nb_epochs=0, nb_batches=128)

        model.herding_indexes = []
        model._task_size = task_info['max_class'] - task_info['min_class']
        model._n_classes += model._task_size


        model.data_memory, model.targets_memory, model.herding_indexes, model.class_means = model.build_examplars(
            inc_dataset, model.herding_indexes)

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(model.class_means)
        nn_graph = nbrs.kneighbors_graph(model.class_means).toarray()
        print(nn_graph)


    #_gcn =  GCN(nfeat=features.shape[1],
    #        nhid=args.hidden,
    #        nclass=labels.max().item() + 1,
    #        dropout=args.dropout)

def main_2():
    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    train_idx = _get_dataset(args["dataset"]).class_order
    print(train_idx)
    _device = args["device"][0]
    _rotations_config = args.get("rotations_config", {})
    inc_dataset = factory.get_data(args, train_idx)
    #print(inc_dataset)
    _network = network.BasicNet(
        args["convnet"],
        convnet_kwargs=args.get("convnet_config", {}),
        classifier_kwargs=args.get("classifier_config", {
            "type": "fc",
            "use_bias": True
        }),
        device=_device,
        extract_no_act=True,
        classifier_no_act=False,
        rotations_predictor=bool(_rotations_config)
    )
    model = factory.get_model(args)
    memory, memory_val = None, None
    metric_logger = metrics.MetricLogger(
        inc_dataset.n_tasks, inc_dataset.n_classes, inc_dataset.increments
    )


    for task_id in range(inc_dataset.n_tasks):
        grad, act = None, None
        clipper = None
        if len(model._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(model._multiple_devices)))
            training_network = nn.DataParallel(model._network, model._multiple_devices)
            if model._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = model._network
        if task_id == 0:
            model._data_memory, model._targets_memory = None, None
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
            print(task_info)
            model._metrics = collections.defaultdict(float)

            model.set_task_info(task_info)
            # ---------------
            # 1. Prepare Task
            # ---------------
            model.eval()
            model.before_task(train_loader, val_loader if val_loader else test_loader)

            # -------------
            # 2. Train Task
            # -------------

            prog_bar = tqdm(
                train_loader,
                disable=model._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                model._optimizer.zero_grad()
                loss = model._forward_loss_1(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    gradcam_grad=grad,
                    gradcam_act=act
                )
                loss.backward()
                model._optimizer.step()

                if clipper:
                    training_network.apply(clipper)

                #model._print_metrics(prog_bar, i, epoch=1, nb_epochs=0)

            if model._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                model._network.eval()
                model._data_memory, model._targets_memory, model._herding_indexes, model._class_means = model.build_examplars(
                    model.inc_dataset, model._herding_indexes
                )
                ytrue, ypred = model._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                model._network.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if model._early_stopping and model._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break
            #features, targets = util.extract_features(_network, train_loader)
            #class_length = collections.defaultdict()
            #features_classi = collections.defaultdict()
            #feature_sumi = collections.defaultdict()
            #for target in targets:
            #    features_classi[target] = []
            #    class_length[target] = 0
            #    feature_sumi[target] = np.zeros((64,), dtype=np.float64)
            #for feature, target in zip(features, targets):
            #    features_classi[target].append(feature.tolist())
            #    class_length[target] += 1
            #    feature_sumi[target] += feature
            #for target in targets:
            #    features_classi[target] = feature_sumi[target] / class_length[target]
            #print(features_classi[23])
            model.herding_indexes = []
            model._task_size = task_info['max_class'] - task_info['min_class']
            model._n_classes += model._task_size

            #feature_class_list = [features_classi[i] for i in set(targets)]
            #feature_class = np.array(feature_class_list)
            #print(len(feature_class_list))
            #nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(feature_class)
            #nn_graph = nbrs.kneighbors_graph(feature_class).toarray() #.reshape(-1, 1)
            #print(nn_graph)
            #model.connect_idx = np.nonzero(nn_graph)
            #print(connect_idx)
            # 每类通过herding策略确定50个范例
            model.data_memory, model.targets_memory, model.herding_indexes, model.class_means = model.build_examplars(
               inc_dataset, model.herding_indexes)

            nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(model.class_means)
            nn_graph = nbrs.kneighbors_graph(model.class_means).toarray() #.reshape(-1, 1)
            model.distances, model.indices = nbrs.kneighbors(model.class_means)
            class_means_ = []
            #print(indices[0])
            for i in model.indices[0]:
                class_means_.append(model.class_means[i])
            #print(np.array(class_means_).shape)
            generate_feature = np.dot(np.array(dis2percent(model.distances[0])).T, np.array(class_means_))
            #print(np.dot(model.class_means[0], generate_feature.T))

            model.connect_idx = np.nonzero(nn_graph)
            #print(list(model.connect_idx)[1].reshape(50,50))
            #print(np.array(model.connect_idx).reshape(50,50))
            memory = model.data_memory, model.targets_memory # , model.connect_idx
            memory_val = model.get_val_memory()

        #else:
        if task_id==1:
            #print(model.indices)
            model._data_memory, model._targets_memory = None, None
            task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)
            print(task_info)
            features, targets = util.extract_features(_network, train_loader)
            print(len(features))
            _data_memory, _targets_memory, _herding_indexes, _class_means = model.build_examplars(
                inc_dataset, model.herding_indexes)
            print(_class_means.shape)





            #print(len(model.herding_indexes))





    #print(len(train_loader))


    #print(features, targets)
    #class_means = np.zeros((_n_classes=100, _network.features_dim))

    # print(len(features_classi[51][0]))
    # print(len(features_classi[51]))



    #print(set(targets))
    #for key, value in features_classi.items():
    #    print(key)

    #print(_network.features_dim)


from train import train
import os
import sys

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
        
def main():
    args = parser_1.get_parser().parse_args()
    args = vars(args)  # Converting argparse Namespace to a dict.
    print(args)

    if args["seed_range"] is not None:
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
        print("Seed range", args["seed"])
        
    init_cls = 0 if args ["initial_increment"] == None else args["initial_increment"] 
    logs_name = "logs/{}/{}/{}/{}".format(args["model"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/epochs{}_lr{}_lrdecay{}_weightdecay{}_scheduling{}".format(
        args["model"],
        args["dataset"],
        init_cls,
        args["increment"],
        
        args["epochs"],
        args["lr"],
        str(args["lr_decay"]),
        args["weight_decay"],
        args["scheduling"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s]: %(message)s', datefmt='%Y-%m-%d:%H:%M:%S',
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    print_args(args)
    # train(args)
    for _ in train(args):  # `train` is a generator in order to be used with hyperfind.
        pass

















if __name__ == '__main__':
    main()