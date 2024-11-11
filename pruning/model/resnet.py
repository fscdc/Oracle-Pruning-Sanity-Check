from torchvision.models import resnet18 as resnet18_imagenet
from torchvision.models import resnet34 as resnet34_imagenet
from torchvision.models import resnet50 as resnet50_imagenet
from torchvision.models import resnet101 as resnet101_imagenet
from torchvision.models import resnet152 as resnet152_imagenet


def resnet18(*args, **kwargs):
    return resnet18_imagenet(pretrained=True)


def resnet34(*args, **kwargs):
    return resnet34_imagenet(pretrained=True)


def resnet50(*args, **kwargs):
    return resnet50_imagenet(pretrained=True)


def resnet101(*args, **kwargs):
    return resnet101_imagenet(pretrained=True)


def resnet152(*args, **kwargs):
    return resnet152_imagenet(pretrained=True)
