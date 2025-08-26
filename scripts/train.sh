# python scripts/train.py --arch unet --encoder tu-resnet50x4_clip
# python scripts/train.py --arch unet --encoder se_resnet152
# python scripts/train.py --arch unet --encoder timm-efficientnet-b0

# python scripts/train.py --arch unet --encoder mobileone_s2
# python scripts/train.py --arch unet --encoder mobileone_s3
# python scripts/train.py --arch segformer --encoder mit_b5
python scripts/train.py --arch unet --encoder timm-efficientnet-b2
python scripts/train.py --arch unet --encoder timm-efficientnet-b5
# python scripts/train.py --arch unet --encoder tu-convnextv2_base

# python scripts/train.py --arch unet --encoder mobileone_s0
# python scripts/train.py --arch unet --encoder mobileone_s1


# python scripts/train.py --arch segformer --encoder mit_b1
# python scripts/train.py --arch segformer --encoder mit_b2



# python scripts/train.py --arch unet --encoder timm-skresnet18
# python scripts/train.py --arch unet --encoder timm-skresnet34
# python scripts/train.py --arch unet --encoder timm-skresnext50_32x4d

# python scripts/train.py --arch segformer --encoder mit_b0

# python scripts/train.py --arch dpt --encoder tu-vit_tiny_patch16_224.augreg_in21k
# python scripts/train.py --arch dpt --encoder tu-vit_small_patch16_224.augreg_in21k
# python scripts/train.py --arch dpt --encoder tu-vit_base_patch16_224.augreg_in21k

# python scripts/train.py --arch unet --encoder resnet34
# python scripts/train.py --arch unet --encoder se_resnet50
# python scripts/train.py --arch steered_cnn --encoder base


# python scripts/train.py --arch maskrcnn --encoder resnet50