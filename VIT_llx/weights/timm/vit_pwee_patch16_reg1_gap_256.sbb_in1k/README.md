---
tags:
- image-classification
- timm
- transformers
library_name: timm
license: apache-2.0
datasets:
- imagenet-1k
---
# Model card for vit_pwee_patch16_reg1_gap_256.sbb_in1k

A Vision Transformer (ViT) image classification model. This is a `timm` specific variation of the architecture with registers, global average pooling.

There are a number of models in the lower end of model scales that originate in `timm`:

| variant | width | mlp width (mult) | heads | depth | timm orig |
| ------- | ----- | ---------------- | ----- | ----- | ---- |
| tiny | 192 | 768 (4) | 3 | 12 | n |
| wee | 256 | 1280 (5) | 4 | 14 | y |
| pwee | 256 | 1280 (5) | 4 | 16 (parallel) | y |
| small | 384 | 1536 (4) | 6 | 12 | n |
| little | 320 | 1792 (5.6) | 5 | 14 | y |
| medium | 512 | 2048 (4) | 8 | 12 | y |
| mediumd | 512 | 2048 (4) | 8 | 20 | y |
| betwixt | 640 | 2560 (4) | 10 | 12 | y |
| base | 768 | 3072 (4) | 12 | 12 | n |

Trained on ImageNet-1k in `timm` using recipe template described below.

Recipe details:
 * Searching for better baselines. Influced by Swin/DeiT/DeiT-III but w/ increased weight decay, moderate (in12k) to high (in1k) augmentation. Layer-decay used for fine-tune. Some runs used BCE and/or NAdamW instead of AdamW.
 * See [train_hparams.yaml](./train_hparams.yaml) for specifics of each model.


## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 15.3
  - GMACs: 3.8
  - Activations (M): 10.6
  - Image size: 256 x 256
- **Papers:**
  - Vision Transformers Need Registers: https://arxiv.org/abs/2309.16588
  - An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale: https://arxiv.org/abs/2010.11929v2
- **Dataset:** ImageNet-1k
- **Original:** https://github.com/huggingface/pytorch-image-models

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('vit_pwee_patch16_reg1_gap_256.sbb_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_pwee_patch16_reg1_gap_256.sbb_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 256, 16, 16])
    #  torch.Size([1, 256, 16, 16])
    #  torch.Size([1, 256, 16, 16])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_pwee_patch16_reg1_gap_256.sbb_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 257, 256) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
| model | top1 | top5 | param_count | img_size |
| -------------------------------------------------- | ------ | ------ | ----------- | -------- |
| [vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_mediumd_patch16_reg4_gap_256.sbb_in12k_ft_in1k) | 86.202 | 97.874 | 64.11 | 256 |
| [vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_256.sbb_in12k_ft_in1k)  | 85.418 | 97.480 | 60.4 | 256 |
| [vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k)  | 84.930 | 97.386 | 38.88 | 256 |
| [vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_mediumd_patch16_rope_reg1_gap_256.sbb_in1k)  | 84.322 | 96.812 | 63.95 | 256 |            
| [vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_betwixt_patch16_rope_reg4_gap_256.sbb_in1k)  | 83.906 | 96.684 | 60.23 | 256 |
| [vit_base_patch16_rope_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_base_patch16_rope_reg1_gap_256.sbb_in1k)  | 83.866 | 96.67 | 86.43 | 256 |
| [vit_medium_patch16_rope_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_medium_patch16_rope_reg1_gap_256.sbb_in1k)  | 83.81 | 96.824 | 38.74 | 256 |
| [vit_little_patch16_reg1_gap_256.sbb_in12k_ft_in1k](https://huggingface.co/timm/vit_little_patch16_reg1_gap_256.sbb_in12k_ft_in1k)  | 83.774 | 96.972 | 22.52 | 256 |
| [vit_betwixt_patch16_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg4_gap_256.sbb_in1k)  | 83.706 | 96.616 | 60.4 | 256 |
| [vit_betwixt_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_betwixt_patch16_reg1_gap_256.sbb_in1k)  | 83.628 | 96.544 | 60.4 | 256 |
| [vit_medium_patch16_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_medium_patch16_reg4_gap_256.sbb_in1k)  | 83.47 | 96.622 | 38.88 | 256 |
| [vit_medium_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_medium_patch16_reg1_gap_256.sbb_in1k)  | 83.462 | 96.548 | 38.88 | 256 |
| [vit_little_patch16_reg4_gap_256.sbb_in1k](https://huggingface.co/timm/vit_little_patch16_reg4_gap_256.sbb_in1k)  | 82.514 | 96.262 | 22.52 | 256 |
| [vit_wee_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_wee_patch16_reg1_gap_256.sbb_in1k)  | 80.258 | 95.360 | 13.42 | 256 |
| [vit_pwee_patch16_reg1_gap_256.sbb_in1k](https://huggingface.co/timm/vit_pwee_patch16_reg1_gap_256.sbb_in1k)  | 80.072 | 95.136 | 15.25 | 256 |

## Citation
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```
```bibtex
@article{darcet2023vision,
  title={Vision Transformers Need Registers},
  author={Darcet, Timoth{'e}e and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv preprint arXiv:2309.16588},
  year={2023}
}
```
```bibtex
@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={ICLR},
  year={2021}
}
```
