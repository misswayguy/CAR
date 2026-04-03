# CAR: Confusion-Aware Spectral Regularizer for Long-Tailed Recognition

> **[CVPR 2026]** Confusion-Aware Spectral Regularizer for Long-Tailed Recognition

## Abstract

Long-tailed image classification remains a long-standing challenge, as real-world data typically follow highly imbalanced distributions where a few head classes dominate and many tail classes contain only limited samples. This imbalance biases feature learning toward head categories and leads to significant degradation on rare classes. Although recent studies have proposed re-sampling, re-weighting, and decoupled learning strategies, the improvement on the most underrepresented classes still remains marginal compared with overall accuracy.

In this work, we present a confusion-centric perspective for long-tailed recognition that explicitly focuses on worst-class generalization. We first establish a new theoretical framework of class-specific error analysis, which shows that the worst-class error can be tightly upper-bounded by the spectral norm of the frequency-weighted confusion matrix and a model-dependent complexity term. Guided by this insight, we propose the **C**onfusion-**A**ware Spectral **R**egularizer (**CAR**) that minimizes the spectral norm of the confusion matrix during training to reduce inter-class confusion and enhance tail-class generalization. To enable stable and efficient optimization, **CAR** integrates a Differentiable Confusion Matrix Surrogate and an EMA-based Confusion Estimator to maintain smooth and low-variance estimates across mini-batches.

Extensive experiments across multiple long-tailed benchmarks demonstrate that **CAR** substantially improves both worst-class accuracy and overall performance. When combined with ConCutMix augmentation, **CAR** consistently surpasses existing state-of-the-art long-tailed learning methods under both the training-from-scratch setting (by **2.37% ~ 4.83%**) and the fine-tuning-from-pretrained setting (by **2.42% ~ 4.17%**) across **ImageNet-LT, CIFAR100-LT, and iNaturalist** datasets.

<p align="center">
  <img src="fig.png" width="900"/>
</p>

## What is in this repository?

We provide the training code for our **CAR** and the baselines.

This repository includes:

- training code for **CAR**
- training code for baseline methods
- implementations of multiple long-tailed learning strategies and augmentation variants
- utilities for analysis and visualization

## Requirements

Please install the required packages with:

```bash
pip install -r requirements.txt
