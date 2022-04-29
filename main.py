import config as c
import torch
from train import train
from utils import load_datasets, make_dataloaders
from torchvision.datasets import MNIST
from torchvision import transforms

if c.class_name == "MNIST":
    mnist_path = c.dataset_path
    Transform = transforms.Compose([transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Resize((448,448))])
    train_set = MNIST(root=mnist_path, train=True, download=False, transform=Transform)
    test_set = MNIST(root=mnist_path, train=False, download=False, transform=Transform)

    index = torch.zeros(len(train_set.data), dtype=torch.bool)
    train_labels = []
    train_label = 8
    train_labels.append(train_label)

    for i in train_labels:
        index |= (train_set.targets == i)
    train_set.targets[index] = 0
    train_set.data, train_set.targets = train_set.data[index], train_set.targets[index]
    train_pct = c.train_pct
    num_train = int(len(train_set.data) / 100 * train_pct)
    train_set.data, train_set.targets = train_set.data[:num_train], train_set.targets[:num_train]

    index = torch.zeros(len(test_set.data), dtype=torch.bool)
    for i in train_labels:
        index |= (test_set.targets == i)
    test_set.targets[index] = 0
    test_set.targets[~index] = 1

else:
    train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    train_pct = c.train_pct
    num_train = int(len(train_set) / 100 * train_pct)
    train_set.samples = train_set.samples[:num_train]
    train_set.imgs = train_set.imgs[:num_train]
    train_set.targets = train_set.targets[:num_train]

train_loader, test_loader = make_dataloaders(train_set, test_set)
model = train(train_loader, test_loader)
