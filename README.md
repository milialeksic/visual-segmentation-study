# Semantic Segmentation for Visual Computing

**Milica Aleksic **  
June 2025

## Abstract

This project compares semantic image segmentation models on the Oxford-IIIT Pet and Cityscapes datasets. The models are PyTorch's FCN-ResNet50 and a transformer-based SegFormer implementation. Experiments examine pretrained weights, encoder freezing, fine-tuning, and learning-rate choices. Performance is assessed using training and validation loss and mean Intersection over Union (mIoU).

## 1. Introduction

Semantic segmentation assigns a class label to every pixel in an image, enabling detailed scene understanding. Unlike image classification, which produces one label per image, segmentation produces a map of object categories and boundaries. It is useful in applications such as autonomous driving and medical image analysis.

This study explores how architecture design and training strategy affect segmentation performance. We compare FCN-ResNet50 and SegFormer, including training from scratch, transfer learning with pretrained weights, freezing parts of a network, and fine-tuning. We also evaluate the effect of changing the learning rate.

## 2. Theoretical foundations

### 2.1 Image semantic segmentation

Semantic segmentation predicts a category for each pixel in an image, providing spatially detailed information about objects and their boundaries.

### 2.2 Pretraining and fine-tuning

Pretraining trains a model on a large dataset so that it learns general visual features. Fine-tuning adapts those features to a smaller, task-specific dataset. This can improve performance and reduce training time compared with training from scratch, particularly when task-specific data is limited.

## 3. Datasets and experimental setup

### 3.1 Datasets

We use two datasets:

- **Oxford-IIIT Pet:** Pet images with three segmentation classes: pet, border, and background.
- **Cityscapes:** Street scenes with 19 classes used for training. Pixels labeled 255 are ignored during training and evaluation.

The report's experiments convert labels to zero-based indices where applicable, while retaining 255 as the ignore label.

### 3.2 Models

- **FCN-ResNet50:** Evaluated from scratch and with pretrained weights.
- **SegFormer:** A transformer-based segmentation model implemented for the project. Evaluated from scratch and with Cityscapes pretraining; the pretrained model is also fine-tuned on Oxford-IIIT Pet with the encoder either frozen or trainable.

### 3.3 Training configuration

The experiments use the AdamW optimizer, Cross Entropy Loss, and an exponential learning-rate scheduler. The reported batch sizes are 64 for Oxford-IIIT Pet and 16 for Cityscapes. Models are trained for 30–40 epochs, depending on the experiment. The ignore index is 255. The best validation model is saved during training, along with the final model.

## 4. Results and evaluation

### 4.1 Experiment configurations

| Model | Pretrained on | Evaluation/training dataset | Encoder frozen | Learning rate |
|---|---|---|---:|---:|
| FCN-ResNet50 | No | Oxford-IIIT Pet | No | 1 × 10⁻³ |
| FCN-ResNet50 | ImageNet/COCO | Oxford-IIIT Pet | No | 1 × 10⁻³ |
| SegFormer | No | Oxford-IIIT Pet | No | 1 × 10⁻³ |
| SegFormer | No | Cityscapes | No | 1 × 10⁻³ |
| SegFormer | Cityscapes | Oxford-IIIT Pet | No | 1 × 10⁻³ |
| SegFormer | Cityscapes | Oxford-IIIT Pet | No | 5 × 10⁻⁴ |
| SegFormer | Cityscapes | Oxford-IIIT Pet | Yes | 1 × 10⁻³ |
| SegFormer | Cityscapes | Oxford-IIIT Pet | Yes | 5 × 10⁻⁴ |

### 4.2 Training results

The pretrained FCN model achieved the highest reported training mIoU (0.835) and the lowest training loss (0.116). Both FCN variants performed strongly. SegFormer trained from scratch achieved training mIoU values of 0.701 on Oxford-IIIT Pet and 0.632 on Cityscapes. SegFormer with a frozen pretrained encoder performed worst among the Oxford fine-tuning configurations, with training mIoU around 0.565–0.568 and loss around 0.472–0.476.

### 4.3 Validation results

The pretrained FCN model achieved the highest validation mIoU (0.694) and lowest validation loss (0.485). FCN trained from scratch followed with validation mIoU of 0.677. Fine-tuned SegFormer configurations reached approximately 0.664–0.665 mIoU when the encoder was trainable. Freezing the encoder reduced validation mIoU to approximately 0.545–0.547. SegFormer trained on Cityscapes and evaluated without fine-tuning on the very different Oxford-IIIT Pet domain achieved approximately 0.200 mIoU.

| Model/configuration | Learning rate | Train mIoU | Validation mIoU | Train loss | Validation loss |
|---|---:|---:|---:|---:|---:|
| FCN pretrained | 1 × 10⁻³ | 0.835 | 0.694 | 0.116 | 0.485 |
| FCN from scratch | 1 × 10⁻³ | 0.830 | 0.677 | 0.121 | 0.501 |
| SegFormer pretrained, fine-tuned | 1 × 10⁻³ | 0.823 | 0.665 | 0.133 | 0.564 |
| SegFormer pretrained, fine-tuned | 5 × 10⁻⁴ | 0.829 | 0.664 | 0.128 | 0.575 |
| SegFormer pretrained, encoder frozen | 1 × 10⁻³ | 0.568 | 0.547 | 0.472 | 0.517 |
| SegFormer pretrained, encoder frozen | 5 × 10⁻⁴ | 0.565 | 0.545 | 0.476 | 0.517 |
| SegFormer from scratch, Oxford-IIIT Pet | 1 × 10⁻³ | 0.701 | 0.543 | 0.262 | 0.622 |
| SegFormer from scratch, Cityscapes | 1 × 10⁻³ | 0.632 | 0.200 | 0.221 | 1.011 |

The report also describes plots of training/validation loss and mIoU, plus example Oxford-IIIT Pet segmentation masks from SegFormer trained from scratch, fine-tuned, and with a frozen encoder. Those figures are not included in this text-only README.

## 5. Conclusion and discussion

FCN-ResNet50 with pretrained weights achieved the strongest overall validation performance. The from-scratch FCN also performed well on Oxford-IIIT Pet. SegFormer produced competitive results when fine-tuned from Cityscapes pretraining, but did not exceed the FCN results in these experiments.

Freezing the SegFormer encoder substantially reduced performance, suggesting that the encoder needed to adapt to the target domain. Cityscapes-only training transferred poorly to Oxford-IIIT Pet, highlighting the impact of domain and label-set differences. Overall, model architecture, pretraining, fine-tuning, and training configuration all influenced segmentation quality.

### 5.1 Learning rate

For fine-tuned SegFormer models with an unfrozen encoder, the two learning rates (1 × 10⁻³ and 5 × 10⁻⁴) produced similar validation mIoU values (0.665 and 0.664). With the encoder frozen, the values were also similar (0.547 and 0.545), though both were substantially lower. In these experiments, learning-rate choice had less impact than whether the encoder was allowed to adapt.

### 5.2 Questions and answers

#### What other segmentation architectures were covered, and how do they differ from SegFormer?

U-Net is a CNN encoder-decoder architecture. Its skip connections combine low-level and high-level features during upsampling, helping retain spatial details. Swin Transformer uses hierarchical vision transformers and shifted windows to model dependencies efficiently. SegFormer also uses a transformer encoder and is designed for simplicity and scalability, without positional encoding in the formulation discussed in the report.

#### Why use downsampling and upsampling instead of keeping the same resolution?

Downsampling reduces spatial resolution, computational cost, and memory use while allowing the model to build higher-level features with a larger receptive field. Upsampling restores predictions to the image's original resolution so the model can produce a pixel-level segmentation map.

#### What is the purpose of pretraining and fine-tuning, and when are they beneficial?

Pretraining lets a model learn general features from a large dataset; fine-tuning adapts those features to a specific task. This is especially beneficial when task-specific data is small or has limited diversity. It can improve generalization, speed convergence, and reduce overfitting risk.

#### How did pretrained weights affect performance, and what was observed when freezing the encoder?

Pretrained weights improved results in the reported comparisons. Fine-tuning SegFormer with an unfrozen pretrained encoder performed better than training from scratch or freezing the encoder. Freezing caused a significant performance drop, indicating that adaptation to the new dataset and domain was important. The tested learning rates produced only small differences within the same training setup.

## Reproducibility note

This README records the project description and results supplied in the report. It does not include training code, model checkpoints, dataset files, or the figures referenced in the report.
