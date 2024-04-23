from torchvision import datasets, transforms
import json
from os import path
from functools import partial

import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
import pendulum

from model.model import *

yaml = YAML()


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(__file__)))


def _construct_report(config, now, model, result_dict):

    report = {}
    report["name"] = config["name"]
    expt = config["experiment"]
    report["expriment"] = expt
    report["time_start"] = now.to_datetime_string()
    report = report | config["params"]["common"]
    report = report | config["params"]["validate"]

    hits, misses, ytest_ypred_dict = 0, 0, {}
    ground_truth_labels = []
    for k, v in result_dict.items():
        labelclass, predclass = k.split("-")[1], k.split("-")[3]
        if labelclass == predclass:
            hits = hits + v
        else:
            misses = misses + v
        ytest_ypred_dict[f"{labelclass}-{predclass}"] = v

        if not labelclass in ground_truth_labels:
            ground_truth_labels.append(labelclass)

    report["hits"] = hits
    report["misses"] = misses
    report["accuracy"] = hits / (hits + misses)

    perclass_accuracy = {}
    for tl in ground_truth_labels:
        chit, cmiss = 0, 0
        for tp in ground_truth_labels:
            tk = f"{tl}-{tp}"
            if tk in ytest_ypred_dict:
                if tl == tp:
                    chit = chit + ytest_ypred_dict[tk]
                else:
                    cmiss = cmiss + ytest_ypred_dict[tk]
        cacc = chit / (chit + cmiss)
        perclass_accuracy[f"{tl}_acc"] = cacc

    report["per_class_accuracy"] = dict(sorted(perclass_accuracy.items()))
    report["per_class_predictions"] = dict(sorted(ytest_ypred_dict.items()))

    return report


def _read_val():
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    print(ASSETS_FP)
    val_fp = path.join(ASSETS_FP, "datasets", "validation")
    print(val_fp)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    val = datasets.MNIST(val_fp, download=False, train=False, transform=transform)
    print("Done: _read_train()")
    return val


def main():
    now = pendulum.now()
    torch.manual_seed(42)
    device = get_device()

    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    experiment_mode = "model"
    experiment = "baseline"
    experiment_timestamp = "20240423T232211"

    config_baseline_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")
    model_fp = path.join(
        ASSETS_FP,
        "models",
        f"{experiment_mode}-{experiment}-{experiment_timestamp}.pkl",
    )

    results_json_fp = path.join(
        ASSETS_FP, "models", f"report-{experiment}-{experiment_timestamp}.json"
    )

    count_unique_labels = 10
    # Config
    config = None
    with open(config_baseline_fp) as f:
        config = yaml.load(f)
    config_common = config.get("params").get("common")
    config_validate = config.get("params").get("validate")

    # Data Loader
    batch_size = config_validate.get("batch_size")
    shuffle = config_validate.get("shuffle")
    val = _read_val()
    loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle)

    # Load model
    model_hyperparameters = {
        "batch_size": config_validate["batch_size"],
        "input_dim": config_common["input_dim"],
        "hidden_dim": config_common["hidden_dim"],
        "freeze_embeddings": config_common["freeze_embeddings"],
    }

    model = ImageMulticlassClassifierBaseline(
        model_hyperparameters, num_classes=count_unique_labels
    )
    model.load_state_dict(torch.load(model_fp))
    model = model.to(device)

    result_dict = {}
    for batch_inputs_BL, batch_labels_BC in loader:
        batch_inputs_BL = batch_inputs_BL.to(device)
        batch_labels_BC = batch_labels_BC.to(device)
        images = batch_inputs_BL.view(batch_inputs_BL.shape[0], -1)
        output = model.forward(images)
        for l_tsr, o_tsr in zip(batch_labels_BC, output):
            l = l_tsr.to(torch.int32)
            l = l.to("cpu")
            l = int(l)
            o = torch.argmax(o_tsr).to(torch.int32)
            o = int(o)
            res = f"label-{l}-pred-{o}"
            if res in result_dict:
                c = result_dict[res]
                c = c + 1
                result_dict[res] = c
            else:
                result_dict[res] = 1

    j = _construct_report(config, now, model, result_dict)
    with open(results_json_fp, "w") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)
    print(f"Done. Wrote results to {results_json_fp}")


if __name__ == "__main__":
    main()
