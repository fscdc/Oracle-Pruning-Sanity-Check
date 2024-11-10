from transformers import AutoImageProcessor, ViTModel, ViTForImageClassification
from torchvision.models.vision_transformer import (
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    vit_h_14,
)

def vit(*args, **kwargs):
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
    return model

def vit_b_16_tv(*args, **kwargs):
    model = vit_b_16(weights='DEFAULT')
    return model

def vit_b_32_tv(*args, **kwargs):
    model = vit_b_32(weights='DEFAULT')
    return model

def vit_l_16_tv(*args, **kwargs):
    model = vit_l_16(weights='DEFAULT')
    return model

def vit_l_32_tv(*args, **kwargs):
    model = vit_l_32(weights='DEFAULT')
    return model

def vit_h_14_tv(*args, **kwargs):
    model = vit_h_14(weights='DEFAULT')
    return model
    







