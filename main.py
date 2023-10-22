import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from CLGAN import CLGAN_Attack, AdvGAN_Attack
from models import MNIST_target_net, MNISTClassifier
import sys
import argparse
from torch.utils.data import random_split

print(sys.path)

parser = argparse.ArgumentParser(description='CLGAN')

parser.add_argument('--dataset', type=bool, default='MNIST')
args = parser.parse_args()

use_cuda=True
image_nc=1
epochs = 500
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1

if args.dataset == 'MNIST':
    use_cuda=True
    image_nc=1
    epochs = 500
    batch_size = 128
    BOX_MIN = 0
    BOX_MAX = 1

if args.dataset == 'cifar-10':
    use_cuda=True
    image_nc=3
    epochs = 500
    batch_size = 128
    BOX_MIN = 0
    BOX_MAX = 255

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda:0" if (use_cuda and torch.cuda.is_available()) else "cpu")

pretrained_model = "MNIST_target_model.pth"
targeted_model = MNISTClassifier().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
clf_train_dataset, attack_train_dataset = random_split(mnist_dataset, [50000, 10000], generator=torch.Generator().manual_seed(42))
dataloader = DataLoader(attack_train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True)
CLGAN = CLGAN_Attack(device,
                          targeted_model,
                          model_num_labels,
                          image_nc,
                          BOX_MIN,
                          BOX_MAX)

CLGAN.train(dataloader, epochs)