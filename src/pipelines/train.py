from torchvision import datasets, transforms
from os import path

import json
import torch
from torch.utils.data import DataLoader
from ruamel.yaml import YAML
import pendulum

from model.model import *

yaml = YAML()


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(__file__)))


def _construct_report(config, losses, now, later, model):
    report = {}
    report["name"] = config["name"]
    report["expriment"] = config["experiment"]
    report = report | config["params"]["common"]
    report = report | config["params"]["train"]

    report["pipeline_time_start"] = now.to_datetime_string()
    report["pipeline_time_end"] = later.to_datetime_string()
    report["training_time_taken_minutes"] = (later - now).in_minutes()
    report["training_loss_curve"] = str(losses)
    report["model_architecture"] = str(model.eval())
    report["model_count_parameters"] = count_parameters(model)
    return report


def _read_train():
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    print(ASSETS_FP)
    train_fp = path.join(ASSETS_FP, "datasets", "train")
    print(train_fp)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train = datasets.MNIST(train_fp, download=False, train=True, transform=transform)
    print("Done: _read_train()")
    return train


def main():

    now = pendulum.now()
    model_timestamp = now.format("YYYYMMDDTHHmmss")
    torch.manual_seed(42)
    device = get_device()

    experiment = "baseline"
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")

    config_fp = path.join(ASSETS_FP, "config", f"config-{experiment}.yaml")
    model_fp = path.join(
        ASSETS_FP, "models", f"model-{experiment}-{model_timestamp}.pkl"
    )
    model_json_fp = path.join(
        ASSETS_FP, "models", f"modelprofile-{experiment}-{model_timestamp}.json"
    )
    count_unique_labels = 10
    # Config
    config = None
    with open(config_fp) as f:
        config = yaml.load(f)
    config_common = config.get("params").get("common")
    config_train = config.get("params").get("train")
    # print(config_train)

    # Data Loader
    batch_size = config_train.get("batch_size")
    shuffle = config_train.get("shuffle")
    train = _read_train()
    loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

    # Model
    model_hyperparameters = {
        "batch_size": config_train["batch_size"],
        "input_dim": config_common["input_dim"],
        "hidden_dim": config_common["hidden_dim"],
        "freeze_embeddings": config_common["freeze_embeddings"],
    }
    model = ImageMulticlassClassifierBaseline(
        model_hyperparameters, num_classes=count_unique_labels
    )
    model = model.to(device)

    # Train model
    learning_rate = config_train["learning_rate"]
    momentum = config_train["momentum"]
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    num_epochs = config_train["epochs"]
    # Train
    epoch_and_loss_list = [["epoch", "loss"]]
    for epoch in range(num_epochs):
        print(f"epoch={epoch}")
        epoch_loss = train_epoch(ce_loss_function, optimizer, model, loader, device)
        print(f"epoch={epoch}, epoch_loss={epoch_loss}")
        epoch_and_loss_list.append([epoch, float(epoch_loss)])
    later = pendulum.now()

    # Save trained model & environment
    torch.save(model.state_dict(), model_fp)
    j = _construct_report(
        config,
        epoch_and_loss_list,
        now,
        later,
        model,
    )
    with open(model_json_fp, "w") as f:
        json.dump(j, f, ensure_ascii=False, indent=4)
    print(f"Done. Timestamp = {model_timestamp}")


if __name__ == "__main__":
    main()
