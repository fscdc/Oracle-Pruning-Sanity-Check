import torch
import torch.nn as nn


class View(nn.ReLU):
    def forward(self, x):
        return x.view(x.size(0), -1)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(84, 10)
        self.view = View()

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = self.view(output)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        return output

class LeNet5_Linear(nn.Module):
    def __init__(self):
        super(LeNet5_Linear, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.act1 = nn.Identity()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.act2 = nn.Identity()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.act3 = nn.Identity()
        self.fc1 = nn.Linear(120, 84)
        self.act4 = nn.Identity()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.act1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.act2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.act3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.act4(output)
        output = self.fc2(output)
        return output


class LeNet5_Wider_Linear(nn.Module):
    """10x num of filters in conv learnable_layers"""

    def __init__(self):
        super(LeNet5_Wider_Linear, self).__init__()

        self.conv1 = nn.Conv2d(1, 60, kernel_size=(5, 5))
        self.act1 = nn.Identity()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(60, 160, kernel_size=(5, 5))
        self.act2 = nn.Identity()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(160, 1200, kernel_size=(5, 5))
        self.act3 = nn.Identity()
        self.fc1 = nn.Linear(1200, 84)
        self.act4 = nn.Identity()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.act1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.act2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.act3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.act4(output)
        output = self.fc2(output)
        return output


class LeNet5_Wider_Linear_NoMaxPool(nn.Module):
    """Use more filters for the first two conv learnable_layers."""

    def __init__(self):
        super(LeNet5_Wider_Linear_NoMaxPool, self).__init__()

        self.conv1 = nn.Conv2d(1, 100, kernel_size=(3, 3), stride=2)  # 14x14
        self.act1 = nn.Identity()
        self.conv2 = nn.Conv2d(100, 100, kernel_size=(3, 3), stride=2)  # 7x7
        self.act2 = nn.Identity()
        self.conv3 = nn.Conv2d(100, 100, kernel_size=(3, 3), stride=2)  # 3x3
        self.act3 = nn.Identity()
        self.fc1 = nn.Linear(900, 84)
        self.act4 = nn.Identity()
        self.fc2 = nn.Linear(84, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.act1(output)
        output = self.conv2(output)
        output = self.act2(output)
        output = self.conv3(output)
        output = self.act3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.act4(output)
        output = self.fc2(output)
        return output


class LeNet5_Mini(nn.Module):
    def __init__(self):
        super(LeNet5_Mini, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        return output


class LeNet5_Mini_w20(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_w20, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(20, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        return output

class LeNet5_Mini_w30(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_w30, self).__init__()

        self.conv1 = nn.Conv2d(1, 30, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(30, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        return output

class LeNet5_Mini_w40(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_w40, self).__init__()

        self.conv1 = nn.Conv2d(1, 40, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(40, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        return output

class LeNet5_Mini_d6(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_d6, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        output = self.relu5(output)
        output = self.fc3(output)
        return output

class LeNet5_Mini_d7(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_d7, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)
        self.relu6 = nn.ReLU()
        self.fc4 = nn.Linear(10, 10)


    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        output = self.relu5(output)
        output = self.fc3(output)
        output = self.relu6(output)
        output = self.fc4(output)
        return output

class LeNet5_Mini_d8(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_d8, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)
        self.relu6 = nn.ReLU()
        self.fc4 = nn.Linear(10, 10)
        self.relu7 = nn.ReLU()
        self.fc5 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        output = self.relu5(output)
        output = self.fc3(output)
        output = self.relu6(output)
        output = self.fc4(output)
        output = self.relu7(output)
        output = self.fc5(output)
        return output

class LeNet5_Mini_d9(nn.Module):
    def __init__(self):
        super(LeNet5_Mini_d9, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(10, 10)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.relu5 = nn.ReLU()
        self.fc3 = nn.Linear(10, 10)
        self.relu6 = nn.ReLU()
        self.fc4 = nn.Linear(10, 10)
        self.relu7 = nn.ReLU()
        self.fc5 = nn.Linear(10, 10)
        self.relu8 = nn.ReLU()
        self.fc6 = nn.Linear(10, 10)

    def forward(self, img):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = self.conv3(output)
        output = self.relu3(output)
        feature = output.view(output.size(0), -1)
        output = self.fc1(feature)
        output = self.relu4(output)
        output = self.fc2(output)
        output = self.relu5(output)
        output = self.fc3(output)
        output = self.relu6(output)
        output = self.fc4(output)
        output = self.relu7(output)
        output = self.fc5(output)
        output = self.relu8(output)
        output = self.fc6(output)
        return output

def lenet5(**kwargs):
    return LeNet5()


def lenet5_mini(**kwargs):
    return LeNet5_Mini()


def lenet5_mini_w20(**kwargs):
    return LeNet5_Mini_w20()


def lenet5_mini_w30(**kwargs):
    return LeNet5_Mini_w30()


def lenet5_mini_w40(**kwargs):
    return LeNet5_Mini_w40()


def lenet5_mini_d6(**kwargs):
    return LeNet5_Mini_d6()


def lenet5_mini_d7(**kwargs):
    return LeNet5_Mini_d7()


def lenet5_mini_d8(**kwargs):
    return LeNet5_Mini_d8()


def lenet5_mini_d9(**kwargs):
    return LeNet5_Mini_d9()

def lenet5_linear(**kwargs):
    return LeNet5_Linear()


def lenet5_wider_linear(**kwargs):
    return LeNet5_Wider_Linear()


def lenet5_wider_linear_nomaxpool(**kwargs):
    return LeNet5_Wider_Linear_NoMaxPool()
