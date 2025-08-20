# python scripts/train.py --arch unet --encoder tu-resnet50x4_clip
python scripts/train.py --arch unet --encoder se_resnet152
python scripts/train.py --arch unet --encoder timm-efficientnet-b0
python scripts/train.py --arch unet --encoder timm-efficientnet-b2
python scripts/train.py --arch unet --encoder timm-efficientnet-b5
python scripts/train.py --arch unet --encoder tu-convnextv2_base




python scripts/train.py --arch segformer --encoder mit_b4
# python scripts/train.py --arch segformer --encoder mit_b0

# python scripts/train.py --arch dpt --encoder tu-vit_tiny_patch16_224.augreg_in21k
# python scripts/train.py --arch dpt --encoder tu-vit_small_patch16_224.augreg_in21k
# python scripts/train.py --arch dpt --encoder tu-vit_base_patch16_224.augreg_in21k

# python scripts/train.py --arch unet --encoder resnet34
# python scripts/train.py --arch unet --encoder se_resnet50
# python scripts/train.py --arch steered_cnn --encoder base


# python scripts/train.py --arch maskrcnn --encoder resnet50