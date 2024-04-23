from torchvision import datasets, transforms
import os
from os import path


def _get_project_dir_folder():
    return path.dirname(path.dirname(path.dirname(__file__)))


def main():
    ASSETS_FP = path.join(_get_project_dir_folder(), "assets")
    print(ASSETS_FP)
    train_fp = path.join(ASSETS_FP, "train")
    print(train_fp)
    validation_fp = path.join(ASSETS_FP, "validation")
    print(validation_fp)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    datasets.MNIST(train_fp, download=True, train=True, transform=transform)
    datasets.MNIST(validation_fp, download=True, train=False, transform=transform)
    print("Done")
    return True


if __name__ == "__main__":
    main()
