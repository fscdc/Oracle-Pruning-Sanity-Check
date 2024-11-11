from .vit import vit, vit_b_16_tv, vit_b_32_tv, vit_l_16_tv, vit_l_32_tv, vit_h_14_tv

model_dict = {
    # ViT model
    'vit': vit,
    'vit_b_16': vit_b_16_tv,
    'vit_b_32': vit_b_32_tv,
    'vit_l_16': vit_l_16_tv,
    'vit_l_32': vit_l_32_tv,
    'vit_h_14': vit_h_14_tv,
}
