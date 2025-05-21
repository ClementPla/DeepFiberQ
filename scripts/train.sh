# python scripts/train.py --arch unetplusplus --encoder se_resnet50

# python scripts/train.py --arch segformer --encoder mit_b4
# python scripts/train.py --arch segformer --encoder mit_b2
# python scripts/train.py --arch segformer --encoder mit_b0
# python scripts/train.py --arch unet --encoder resnet34
# python scripts/train.py --arch unet --encoder se_resnet50
# python scripts/train.py --arch segformer --encoder se_resnet50
# python scripts/train.py --arch segformer --encoder convnext_base
python scripts/train.py --arch unet --encoder convnext_base

# python scripts/train.py --arch dpt --encoder tu-vit_tiny_patch16_224.augreg_in21k
# python scripts/train.py --arch dpt --encoder tu-vit_small_patch16_224.augreg_in21k
# python scripts/train.py --arch dpt --encoder tu-vit_base_patch16_224.augreg_in21k


# python scripts/train.py --arch maskrcnn --encoder resnet50