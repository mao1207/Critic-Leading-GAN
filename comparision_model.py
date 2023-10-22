import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg
from torchvision.models import resnet

# MNIST test

# modelA
class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64*20*20, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)

# modelB
class ModelB(nn.Module):
    def __init__(self):
        super(ModelB, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*18*18, 10)

    def forward(self, x):
        x = self.dropout1(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.softmax(x, dim=1)

# modelC
class ModelC(nn.Module):
    def __init__(self):
        super(ModelC, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim = 1)


# modelD
class ModelD(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ModelD, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return F.softmax(out, dim=1)

# CIFAR10 test
class ResNet32(nn.Module):
    def __init__(self, num_classes):
        super(ResNetModel, self).__init__()
        self.model = resnet.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        self.input_standardized = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.unit_1_0 = nn.Sequential(
            nn.Conv2d(16, 160, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(160, 160, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.unit_last = nn.Sequential(
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.logit = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 10)
        )

    def forward(self, x):
        x = self.input_standardized(x)
        x = self.unit_1_0(x)
        x = self.unit_last(x)
        x = self.logit(x)

        return x


class VGGModel(nn.Module):
    def __init__(self, num_classes):
        super(VGGModel, self).__init__()
        self.model = vgg.vgg16_bn(pretrained=True)
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

