
> **The goal of this project was not to achieve state-of-the-art accuracy.** Instead, the objective was to investigate how different training techniques (handling class imbalance, augmentation, optimization methods, AMP, learning-rate scheduling, etc.) can produce a reasonable classifier using only a relatively small subset (30%) of a highly imbalanced dataset while operating under GPU memory constraints.

That objective should be emphasized throughout the documentation because it explains many of your engineering decisions.

---

# Documentation for Part 1 — CNN Supervised Learning

## 1. Project Overview

This project investigates food image classification using a Convolutional Neural Network (CNN) under realistic computational constraints. The dataset consists of 251 food categories with significant class imbalance and varying image resolutions. Rather than using the entire dataset, only 30% of the available images were selected to reduce computational cost while maintaining the original class distribution.

The primary objective of this supervised learning experiment was not to maximize classification accuracy, but to examine how various practical training strategies—including data augmentation, class imbalance handling, learning-rate scheduling, Automatic Mixed Precision (AMP), and modern optimization methods—can improve learning efficiency when limited training data and GPU memory are available.

This CNN model serves as the supervised learning baseline for later comparison with a Self-Supervised Learning (SSL) approach using the same dataset.

---

# 2. Objectives and Motivation

The project was designed with the following objectives:

* Develop a CNN-based image classifier for food recognition.
* Train the model using only a subset (30%) of the dataset while preserving the original class distribution.
* Investigate the effect of practical deep learning techniques rather than pursuing maximum accuracy.
* Handle severe class imbalance during training.
* Reduce computational requirements to fit GPU memory limitations.
* Produce a supervised learning baseline that can later be compared fairly against a Self-Supervised Learning approach.

The motivation behind selecting only 30% of the dataset was to ensure that both supervised and self-supervised approaches would be trained using the same amount of data, enabling a fair comparison while remaining within hardware constraints.

---

# 3. Overall Workflow

The supervised learning pipeline consists of the following stages:

```
Original Dataset
        │
        ▼
Random Sampling (30%)
        │
        ▼
Maintain Original Class Distribution
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Compute Dataset Mean & Standard Deviation
        │
        ▼
Image Resize (160×160)
        │
        ▼
Data Augmentation (Training Only)
        │
        ▼
Class Weight Calculation
        │
        ▼
Weighted Sampling
        │
        ▼
CNN Training
        │
        ▼
Validation During Training
        │
        ▼
Performance Evaluation
```

Each stage contributes to improving training efficiency, reducing overfitting, or ensuring fair evaluation.

---

# 4. System Architecture

The implemented CNN architecture is inspired by the ResNet family of networks.

Instead of using a standard sequential CNN, the model incorporates residual skip connections that help information flow through the network and alleviate degradation problems in deeper architectures.

The architecture contains:

* Five residual blocks
* Residual skip connections
* Batch Normalization after convolutional layers
* ReLU activation functions
* Dropout (0.5) before the classifier
* Fully connected classification layer with 251 output neurons
* 6,470,180 trainable parameters 

### Design Decisions

### Residual Connections

Residual connections allow gradients to flow more easily during backpropagation, making deeper networks easier to optimize.

### Batch Normalization

Batch normalization was added after convolutional operations to stabilize training and reduce internal covariate shift.

### Dropout

A dropout probability of 0.5 was introduced before the final classifier to reduce overfitting by randomly deactivating neurons during training.

Overall, the architecture balances model complexity with computational efficiency.

### Model Complexity

The implemented CNN contains 6,470,180 trainable parameters, satisfying the project requirement of keeping the model below 10 million parameters. This design balances representational capacity with computational efficiency, allowing the network to learn meaningful image features while remaining suitable for the available GPU memory and training time constraints.
---

# 5. Dataset Description

The project uses a food recognition dataset containing images from 251 food categories.

### Original Dataset

| Split    |  Images |
| -------- | ------: |
| Training | 118,475 |
| Testing  |  11,994 |

Each class contains between:

* minimum: 34 images
* maximum: 656 images

making the dataset highly imbalanced.

The original images also have uncontrolled resolutions, requiring preprocessing before training.

---

## Dataset Sampling

To reduce computational cost while maintaining fairness for later comparison with Self-Supervised Learning, only 30% of the dataset was randomly selected.

A fixed random seed (`random_state = 42`) was used to ensure reproducibility.

The sampled dataset preserves the original class distribution.

Dataset distribution figures (100% vs. 30%) demonstrate that the sampling process maintains the overall data balance across classes.

---

## Final Dataset Split

After sampling:

| Dataset    | Images |
| ---------- | -----: |
| Training   | 26,568 |
| Validation |  8,856 |
| Testing    |  3,485 |

The validation set was extracted from the training data using a 75% / 25% split.

---

# 6. Data Preprocessing

Several preprocessing operations were performed before training.

## Dataset Normalization

The global mean and standard deviation were computed across the dataset:

Mean

```
[0.485, 0.456, 0.406]
```

Standard deviation

```
[0.229, 0.224, 0.225]
```

These statistics were used to normalize every image, ensuring that pixel values are centered and scaled consistently during training.

---

## Image Resizing

All images were resized to

```
160 × 160
```

This size was selected as a compromise between:

* preserving sufficient visual information
* reducing GPU memory usage
* increasing training speed

Since the original dataset contains images with varying resolutions, resizing also standardizes the model input.

---

## Data Augmentation

To improve generalization without increasing dataset size on disk, online data augmentation was applied only during training.

The augmentations include:

* Random crop
* Random brightness adjustment
* Random contrast adjustment
* Random horizontal flip

These transformations generate slightly different versions of images each epoch, helping the model learn more robust visual features.

---

## Class Imbalance Handling

The dataset contains large differences in the number of samples per class.

To reduce bias toward majority classes:

* class frequencies were calculated
* normalized class weights were computed
* a weighted sampler was created for the training DataLoader

This sampling strategy increases the probability of selecting minority-class images during training without modifying the dataset itself.

---

## Data Loading

Training parameters:

Batch size:

```
64
```

Pin memory:

```
Enabled
```

Pinning memory accelerates CPU-to-GPU data transfer during training.

Total batches:

| Split      | Batches |
| ---------- | ------: |
| Train      |     415 |
| Validation |     139 |
| Test       |      55 |

---

# 7. Model Selection and Training Strategy

The CNN was trained for

```
50 epochs
```

This value was chosen to balance computational cost and training time.

### Optimizer

AdamW was selected because it separates weight decay from gradient updates, providing more stable optimization than traditional Adam.

### Loss Function

Cross-Entropy Loss was used because this is a **multi-class classification** problem with 251 output classes.
### Learning Rate Scheduler

A learning-rate scheduler updates the learning rate throughout training, enabling larger updates during early learning and smaller updates as training converges.

Final learning rate:

```
1 × 10⁻⁶
```

### Automatic Mixed Precision (AMP)

Automatic Mixed Precision was incorporated to:

* reduce GPU memory consumption
* accelerate training
* maintain numerical stability using gradient scaling

This allows larger batch sizes and faster execution without noticeably affecting model performance.

### Training Metric

During training and validation, model performance was monitored using Raw Accuracy (micro-averaged accuracy). This metric computes the proportion of correctly classified samples across the entire dataset without averaging performance across individual classes. Raw accuracy provides an intuitive measure of overall classification performance and was used to monitor the model's learning progress throughout the 50 training epochs.
---

# 8. Techniques Used

The implemented training pipeline combines several practical techniques to improve efficiency and learning under limited resources:

* Random dataset sampling (30%)
* Dataset normalization
* Image resizing
* Online data augmentation
* Weighted random sampling
* Residual CNN architecture
* Batch normalization
* Dropout regularization
* AdamW optimization
* Cross-Entropy Loss
* Learning-rate scheduling
* Automatic Mixed Precision (AMP)
* Early stopping (implemented but not triggered)

These techniques collectively improve training stability, computational efficiency, and robustness without increasing model complexity.

---

# 9. Experimental Setup

The CNN was trained using the sampled dataset under GPU memory constraints.

Training configuration:

| Parameter      | Value         |
| -------------- | ------------- |
| Epochs         | 50            |
| Batch Size     | 64            |
| Image Size     | 160×160       |
| Optimizer      | AdamW         |
| Loss           | Cross-Entropy |
| Scheduler      | Yes           |
| AMP            | Yes           |
| Early Stopping | Enabled       |

Training completed in approximately

```
200 minutes
```
### Performance Monitoring: 
* Training metric: Raw Accuracy (micro-averaged accuracy)
* Final evaluation metrics: Precision, Recall, F1-score, and Confusion Matrix

---

# 10. Results and Observations

## Final Training Metrics

Epoch 50

Training Loss

```
3.4648
```

Validation Loss

```
3.9803
```

Training Accuracy

```
22.38%
```

Validation Accuracy

```
19.00%
```

---

## Validation Evaluation

Although raw accuracy was used to monitor learning during training, it is less informative for highly imbalanced datasets. Therefore, Precision, Recall, and F1-score were also reported during the final evaluation to provide a more comprehensive assessment of model performance across the 251 classes.

Precision

```
0.1647
```

Recall

```
0.1898
```

F1 Score

```
0.1668
```

---

## Test Evaluation

Precision

```
0.1917
```

Recall

```
0.2095
```

F1 Score

```
0.1854
```

---

## Training Behaviour

The learning curves indicate that:

* training loss consistently decreases throughout training
* validation loss decreases initially but begins to oscillate after approximately epoch 40
* validation accuracy also slows after epoch 40

This behaviour suggests that although the model continues learning, performance improvements become smaller. The oscillating validation loss may be influenced by the limited training data and the highly imbalanced class distribution.

The implemented early stopping mechanism was never triggered, indicating that the stopping criterion was not met before the maximum number of epochs.

Confusion matrices and per-class F1-score analysis further demonstrate the impact of class imbalance. Classes with more training examples generally achieved higher performance, while minority classes were more difficult to classify accurately.

---

# 11. Strengths of the Implemented Approach

The implemented approach offers several strengths:

* Preserves fair comparison by using the same sampled dataset intended for both supervised and self-supervised experiments.
* Handles severe class imbalance through weighted sampling.
* Employs practical data augmentation to improve generalization.
* Incorporates residual learning and batch normalization for stable optimization.
* Uses Automatic Mixed Precision to reduce GPU memory usage and accelerate training.
* Maintains reproducibility through fixed random sampling.
* Balances computational efficiency with model performance under limited hardware resources.

Although the overall accuracy is modest, the model demonstrates meaningful learning despite the challenging dataset characteristics and restricted training conditions.

---

# 12. Limitations and Challenges

Several limitations were observed during implementation:

* Only 30% of the available training data was used.
* The dataset exhibits significant class imbalance across 251 categories.
* Some classes contain very few training samples.
* Image resizing may remove fine-grained visual details.
* Training was limited to 50 epochs due to computational constraints.
* Validation performance plateaued after approximately epoch 40.
* Minority classes remain considerably more difficult to classify than majority classes.

These limitations should be considered when interpreting the reported performance.

---

# 13. Lessons Learned

This experiment demonstrates that effective engineering decisions can significantly improve training efficiency even when computational resources are limited. Techniques such as weighted sampling, data augmentation, residual learning, learning-rate scheduling, and Automatic Mixed Precision contributed to stable optimization and enabled the model to learn meaningful representations from a reduced dataset.

The results also highlight the practical challenges posed by highly imbalanced datasets, where performance is often constrained by the availability of samples in minority classes. While the implemented strategies helped mitigate these effects, class imbalance continued to influence overall classification performance.

---

# 14. Future Work and Possible Improvements

The following ideas were **not implemented** in this project but could be explored in future work:

* Train using the full dataset to provide more examples for minority classes.
* Experiment with deeper or pre-trained CNN architectures.
* Investigate alternative loss functions specifically designed for imbalanced classification (e.g., focal loss).
* Explore more advanced data augmentation techniques.
* Perform systematic hyperparameter optimization.
* Increase image resolution if additional GPU resources are available.
* Evaluate additional performance metrics and calibration methods.
* Compare the supervised baseline with the planned Self-Supervised Learning pipeline using identical experimental conditions.

---

# 15. Conclusion

This part of the project established a supervised learning baseline for food image classification using a custom CNN architecture inspired by ResNet. The experiment deliberately prioritized computational efficiency and practical training strategies over maximizing predictive accuracy. By training on only 30% of a highly imbalanced dataset, the study demonstrated how techniques such as weighted sampling, online data augmentation, residual connections, Batch Normalization, AdamW optimization, learning-rate scheduling, and Automatic Mixed Precision can be combined to obtain meaningful performance within limited hardware resources.

Although the final accuracy and F1-scores reflect the difficulty of the task, the model successfully learned discriminative features across 251 food categories and provided a solid reference point for the second phase of the project, where its performance will be compared with a Self-Supervised Learning approach under the same experimental conditions.


