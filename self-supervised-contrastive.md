# Self-Supervised Learning Using SimCLR

## 1. Motivation

After establishing the supervised CNN baseline, a Self-Supervised Learning (SSL) approach was investigated using SimCLR.

The objective was not to increase model complexity or achieve state-of-the-art accuracy, but to evaluate whether learning visual representations without labels could improve classification performance when only a limited amount of training data is available.

For a fair comparison with the baseline CNN:

* The same dataset subset was used.
* The same original class distribution was preserved.
* The same ResNet-based encoder architecture was used.

The main difference was the training strategy:

* The baseline CNN learns directly from image-label pairs.
* SimCLR learns general image representations from relationships between augmented views of the same image.

---

# 2. Dataset and Preprocessing

The dataset preparation remained unchanged from the supervised CNN experiment.

Only 30% of the original dataset was randomly selected while preserving the original class distribution.

The same:

* image size
* normalization
* train/validation split
* dataset sampling strategy

were maintained to ensure a fair comparison between supervised and self-supervised approaches.

---

# 3. SimCLR Data Augmentation Strategy

Unlike supervised learning, where augmentation is mainly used as a regularization technique, SimCLR uses augmentation as the main learning signal.

For each input image, two different augmented versions are generated:

```
              Original Image
                    |
        -------------------------
        |                       |
 Augmentation 1          Augmentation 2
        |                       |
       x_i¹                    x_i²
```

These two images represent the same object but appear visually different to the network.

The objective of SimCLR is to learn that:

* augmented views of the same image should have similar representations
* images from different samples should have different representations

---

## Difference Between CNN and SimCLR Augmentation

Although both approaches use image augmentation, their purpose is different.

### Supervised CNN

Augmentation is used to:

* increase data diversity
* reduce overfitting
* improve classification robustness

### SimCLR

Augmentation is used to:

* create positive training pairs
* force the model to learn invariant visual features
* provide the learning signal without labels

Therefore, using exactly the same augmentation pipeline was not optimal.

The augmentation strategy was strengthened for SimCLR by adding:

* stronger color variation
* contrast adjustment
* Gaussian blur

These transformations encourage the encoder to learn features that remain stable despite visual changes.

---

# 4. SimCLR Architecture

To maintain a fair comparison, the same ResNet architecture used in the supervised CNN was selected as the encoder.

The SimCLR model consists of:

```
Image
 |
Encoder (ResNet)
 |
Feature Representation
 |
Projection Head
 |
128-dimensional embedding
```

The projection head contains a small neural network that maps extracted features into a representation space optimized for contrastive learning.

---

## Projection Dimension

The projection head outputs a 128-dimensional vector.

The choice of 128 dimensions is a common design choice in contrastive learning frameworks, including the original SimCLR work.

The purpose of the projection head is:

* The encoder learns useful visual features.
* The projection head learns a space optimized for similarity comparison.
* The learned encoder representation can later be transferred to downstream classification tasks.

The total number of parameters:

```
6,632,352 parameters
```

which remains within the project computational constraints.

---

# 5. Self-Supervised Training Process

During SimCLR training, the DataLoader behavior changes compared with supervised learning.

Instead of returning:

```
(image, label)
```

the DataLoader returns:

```
(augmented_view_1, augmented_view_2)
```

Both images originate from the same original sample but contain different augmentations.

The model learns without using class labels.

---

# 6. Contrastive Loss Function

Since SimCLR is not a classification problem during pretraining, Cross-Entropy Loss is not appropriate.

Instead, the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss was used.

This is the standard contrastive loss used in SimCLR.

The objective is:

* maximize similarity between positive pairs:

```
same image + different augmentation
```

* minimize similarity between negative pairs:

```
different images
```

The model therefore learns meaningful visual representations without explicit labels.

---

# 7. Training Acceleration

To improve training efficiency:

* Automatic Mixed Precision (AMP) was used.
* PyTorch JIT compilation (`torch.compile`) was applied.

The compilation technique optimizes model execution by generating optimized GPU kernels and reducing unnecessary computation overhead.

---

# 8. SimCLR Training Results

The model was trained for:

```
474.3872 minutes
```

Because SimCLR is a representation learning method, classification accuracy is not calculated during pretraining.

Instead, training and validation contrastive loss were monitored.

[Loss Curve Figure]

---

## Loss Curve Analysis

The loss curves show:

* training loss decreases continuously throughout training.
* validation loss follows a similar decreasing trend.
* both curves remain relatively parallel.

This indicates that the model continues learning useful image representations without significant overfitting.

Although small oscillations appear at the beginning of training, the overall trend shows stable convergence.

Early stopping based on validation loss was implemented; however, it was never activated because the validation loss continued improving throughout training.

---

# 9. Feature Representation Visualization

After SimCLR pretraining, intermediate feature maps were visualized to understand what the encoder learned.

The analyzed network blocks represent progressively deeper features:

| Block   | Learned Features    |
| ------- | ------------------- |
| Block 1 | Low-level edges     |
| Block 2 | Textures            |
| Block 3 | Shapes              |
| Block 4 | High-level patterns |

The visualization showed that:

* early layers successfully learned edges and texture information.
* deeper layers captured more complex patterns.
* high-level food structures were more difficult because many food categories share similar appearance and presentation styles.

For example, many food images contain similar backgrounds, plates, and shapes, making fine-grained category separation challenging.

---

# 10. Fine-Tuning for Classification

After SimCLR pretraining, the encoder was transferred to a classification task.

The objective was not only to use pretrained representations but also to adapt the model to food-specific classification details.

The following strategy was applied:

Frozen layers:

* Block 1
* Block 2

Trainable layers:

* Block 3
* Block 4
* Block 5

This approach reduces training time while allowing higher-level features to adapt to the food classification task.

---

# 11. Classification Head

A new classifier head was added:

```
Adaptive Average Pooling
        |
Flatten
        |
Linear Layer
        |
Batch Normalization
        |
ReLU
        |
Dropout
        |
Linear Layer
        |
251 Classes
```

Total parameters:

```
7,481,243 parameters
```

---

# 12. Fine-Tuning Strategy

The optimizer was adjusted to use different learning rates for different layers.

Lower learning rates were applied to pretrained encoder layers, while the classifier head received a larger learning rate.

This prevents destroying useful pretrained representations while allowing the classifier to adapt.

The loss function was changed back to:

```
Cross-Entropy Loss
```

because the task became supervised multi-class classification.

AMP was also used to:

* accelerate training
* reduce GPU memory usage

---

# 13. Fine-Tuning Results

Training completed after:

```
195 minutes
```

## Loss Curve Analysis

[Loss Curve Figure]

Observations:

* Training loss decreased from approximately:

```
5.12 → 3.72
```

showing continuous learning.

* Validation loss decreased from:

```
4.76 → 4.20
```

but improvement slowed after approximately epoch 30–35.

* The gap between training and validation loss increased gradually, reaching approximately 0.48 at the end.

This indicates mild overfitting, but the validation loss never increased significantly, suggesting that the model maintained reasonable generalization.

---

## Accuracy Curve Analysis

[Accuracy Curve Figure]

Training accuracy:

```
3.62% → 18.51%
```

Validation accuracy:

```
7.14% → 14.42%
```

Observations:

* Training accuracy increased almost continuously.
* Validation accuracy improved steadily but slower.
* Both curves began plateauing around epoch 35–40.
* The final difference between training and validation accuracy was approximately 4%.

Compared with the supervised CNN baseline, the SimCLR-based model showed a smaller generalization gap, suggesting improved robustness when training with limited data.

---

# 14. Comparison With Baseline CNN

The supervised CNN learned directly from image-label pairs, while SimCLR first learned general visual representations and then adapted them for classification.

Using the same dataset size:

| Method               | Advantages                                                        | Limitations                                 |
| -------------------- | ----------------------------------------------------------------- | ------------------------------------------- |
| CNN Baseline         | Faster training, simpler pipeline                                 | More prone to overfitting with limited data |
| SimCLR + Fine-tuning | Better feature generalization, stronger pretrained representation | Requires significantly longer training      |

The SimCLR approach required more computational time because:

* each image generates two augmented inputs
* both views pass through the encoder
* contrastive loss performs embedding comparisons

However, the additional pretraining stage resulted in a more generalized feature extractor.

Therefore, under the limited 30% dataset condition, the SimCLR-based approach provided better generalization compared with the supervised CNN baseline.

---

# 15. Limitations

The main limitation of SimCLR was computational cost.

Compared with supervised learning:

* training time was significantly longer.
* memory usage increased due to two image views per sample.
* contrastive similarity calculations added additional computation.

Although SimCLR provided better feature learning, the additional cost may not be justified when very large labeled datasets are available.

---

# 16. Precision, Recall, and F1-score Limitation

Precision, recall, and F1-score were not reported because the final trained checkpoint and prediction outputs were unavailable after the Kaggle runtime session expired.

These metrics require running inference on the evaluation dataset and comparing predicted labels against ground truth labels.

Since the prediction outputs were not saved, these metrics could not be reconstructed retrospectively.

Therefore, evaluation was based on the available:

* training loss
* validation loss
* training accuracy
* validation accuracy

recorded during training.

---

# 17. Final Conclusion

The SimCLR experiment demonstrated that self-supervised representation learning can improve model robustness when working with limited labeled data.

Although the method requires significantly more computational resources than supervised CNN training, the learned representations provided better generalization during downstream classification.

The experiment highlights the trade-off between:

* computational efficiency (supervised CNN)
* representation quality and generalization (SimCLR)

For small datasets with limited labels, self-supervised pretraining represents a promising approach for improving classification performance without requiring additional annotation.


