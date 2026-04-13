# CAR: Confusion-Aware Spectral Regularizer for Long-Tailed Recognition

> **[CVPR 2026 Oral]** Confusion-Aware Spectral Regularizer for Long-Tailed Recognition

## Abstract

Long-tailed image classification remains a long-standing challenge, as real-world data typically follow highly imbalanced distributions where a few head classes dominate and many tail classes contain only limited samples. This imbalance biases feature learning toward head categories and leads to significant degradation on rare classes. Although recent studies have proposed re-sampling, re-weighting, and decoupled learning strategies, the improvement on the most underrepresented classes still remains marginal compared with overall accuracy. In this work, we present a confusion-centric perspective for long-tailed recognition that explicitly focuses on worst-class generalization. We first establish a new theoretical framework of class-specific error analysis, which shows that the worst-class error can be tightly upper-bounded by the spectral norm of the frequency-weighted confusion matrix and a model-dependent complexity term. Guided by this insight, we propose the **C**onfusion-**A**ware Spectral **R**egularizer (**CAR**) that minimizes the spectral norm of the confusion matrix during training to reduce inter-class confusion and enhance tail-class generalization. To enable stable and efficient optimization, **CAR** integrates a Differentiable Confusion Matrix Surrogate and an EMA-based Confusion Estimator to maintain smooth and low-variance estimates across mini-batches. Extensive experiments across multiple long-tailed benchmarks demonstrate that **CAR** substantially improves both worst-class accuracy and overall performance. When combined with ConCutMix augmentation, **CAR** consistently surpasses existing state-of-the-art long-tailed learning methods under both the training-from-scratch setting (by **2.37% ~ 4.83%**) and the fine-tuning-from-pretrained setting (by **2.42% ~ 4.17%**) across **ImageNet-LT, CIFAR100-LT, and iNaturalist** datasets.

<p align="center">
  <img src="fig.png" width="900"/>
</p>

## What is in this repository?

We provide the training code for our **CAR** and the baselines.

This repository includes:

- training code for **CAR**
- training code for baseline methods
- utilities for analysis and visualization

## Requirements


Please install the required packages with:

```bash
pip install -r requirements.txt
```

## Training


### Example: Training with ViT-S

Run the following command to train CAR with a ViT-S backbone:

```bash
CUDA_VISIBLE_DEVICES=0 python train_car_tail.py \
  --train-dir /your_train_dataset_path \
  --val-dir /your_test_dataset_path \
  --model vit_small_patch16_224 \
  --img-size 224 \
  --batch-size 128 \
  --epochs 100 \
  --opt adamw \
  --lr 5e-5 \
  --weight-decay 0.05 \
  --no-pretrained \
  --init /your_vit_small_checkpoint_path \
  --sched cosine \
  --warmup-epochs 5 \
  --min-lr 1e-6 \
  --use-wj \
  --spec-lambda 0.3 \
  --spec-beta 0.5 \
  --spec-temp 1 \
  --hmt \
  --head-th 100 \
  --tail-th 20 \
  --save ./weights/ckpt.pth\
  --wj-norm \
  --wj-min 1e-3 \
  --r0 0.5
```

### Example: Training baseline with ViT-S

Run the following command to train the baseline model with LDAM loss:

```bash
CUDA_VISIBLE_DEVICES=5 python train_baseline_tail.py \
  --train-dir /your_train_dataset_path \
  --val-dir /your_test_dataset_path \
  --model vit_small_patch16_224 \
  --img-size 224 \
  --batch-size 32 \
  --epochs 30 \
  --lr 3e-4 \
  --no-pretrained \
  --init /your_vit_small_checkpoint_path \
  --head-th 100 \
  --tail-th 20 \
  --save ./weights/ckpt.pth\

