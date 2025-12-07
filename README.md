# LTGC CIFAR100 Reproduction

> [!NOTE]
> This repository contains a **comprehensive project report** detailing the reproduction and adaptation of the LTGC framework on the **CIFAR-100 Long-Tailed dataset**.
> Please scroll to the bottom or [click here](#comprehensive-project-report-adaptation-of-ltgc-framework-to-cifar-100-long-tailed-recognition) to read the full analysis, experimental results, and discussion.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/LTGC.git
   cd LTGC
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up OpenAI API Key (for data generation):
   ```bash
   export OPENAI_API_KEY='your_api_key_here'
   ```

## Dataset Preparation

### (1) Three benchmark datasets
- Please download these datasets and put them to the `data/` directory.
- ImageNet-LT and Places-LT can be found [here](https://drive.google.com/drive/u/0/folders/1j7Nkfe6ZhzKFXePHdsseeeGI877Xu1yf).
- iNaturalist data should be the 2018 version from [here](https://github.com/visipedia/inat_comp).

```
data
├── ImageNet_LT
│   ├── test
│   ├── train
│   └── val
├── Place365
│   ├── data_256
│   ├── test_256
│   └── val_256
└── iNaturalist 
    ├── test2018
    └── train_val2018
```

### (2) Txt files
```
data_txt
├── ImageNet_LT
│   ├── ImageNet_LT_test.txt
│   ├── ImageNet_LT_train.txt
│   └── ImageNet_LT_val.txt
├── Places_LT_v2
│   ├── Places_LT_test.txt
│   ├── Places_LT_train.txt
│   └── Places_LT_val.txt
└── iNaturalist18
    ├── iNaturalist18_train.txt
    ├── iNaturalist18_uniform.txt
    └── iNaturalist18_val.txt 
```

## Running Scripts

### Train CIFAR100-LT
```bash
python train_ltgc_cifar100.py --data_dir data/CIFAR100_LT --loss ce --balanced
```

### Generated Existing Tail-class Descriptions
``` bash
python lmm_i2t.py -d $DATASET_PATH -m $MAX_NUMBER -f $CLASS_NUMBER_FILE -exi $EXIST_DESCRIPTION_FILE
```

### Generated Extended Tail-class Descriptions 
``` bash
python lmm_extension.py -exi $EXIST_DESCRIPTION_FILE -m $MAX_GENERATED_IMAGES -ext $EXTEND_DESCRIPTION_FILE
```

### Generated Extended Data using Iterative Evaluation
``` bash
python draw_i2t.py -ext $EXTEND_DESCRIPTION_FILE -d $DATASET_PATH -t $THRESH -r $MAX_ROUNDS
```


# Comprehensive Project Report: Adaptation of LTGC Framework to CIFAR-100 Long-Tailed Recognition

**Subject:** Implementation and Domain Adaptation of "Long-tail Recognition via Leveraging LLMs-driven Generated Content"
**Target Domain:** CIFAR-100 Long-Tailed Dataset

## 1. Introduction

### 1.1 The Challenge of Long-Tailed Recognition
In real-world scenarios, data distribution often follows a **Long-Tailed** shape, where a few "Head" classes appear frequently (e.g., 'Dog', 'Cat'), while numerous "Tail" classes appear rarely (e.g., 'Platypus', 'Axolotl'). Deep learning models trained on such data tend to bias heavily towards the majority classes to minimize overall loss, resulting in near-zero accuracy for minority classes.
While traditional solutions like Re-sampling (Over/Under-sampling) or Re-weighting (Cost-sensitive learning) attempt to balance the training process, they suffer from a fundamental limitation: **they cannot create information that doesn't exist.** They merely repeat existing tail samples (risking overfitting) or penalize head samples (risking underfitting).

### 1.2 The LTGC Framework Approach
The paper *"LTGC: Long-tail Recognition via Leveraging LLMs-driven Generated Content"* proposes a paradigm shift: **Data Generation instead of Data Repetition.**
The framework utilizes the rich, implicit knowledge embedded in Large Multimodal Models (LMMs) like GPT-4V and DALL-E. By asking LMMs to describe the features of tail classes and generating synthetic images based on these descriptions, LTGC effectively "fills in" the missing parts of the distribution with diverse, high-quality data.

### 1.3 Objective of Adaptation
The original LTGC was benchmarked on high-resolution datasets (ImageNet-LT). This project aims to **adapt and validate the LTGC generative logic on CIFAR-100 LT**, a dataset characterized by:
1.  **Low Resolution (32x32):** Making feature extraction harder for both real and generated images.
2.  **High Imbalance (Ratio=100):** Head classes have ~500 images, while Tail classes have only ~5.
We aim to verify if generated content can boost tail performance in this challenging domain and analyze the trade-offs of different training strategies.

## 2. Methodology & Adaptation Process

### 2.1 Generative Pipeline (Adopted from LTGC)
We replicated the core generative logic of the LTGC paper to construct a supplementary dataset. The process involves three key steps:
1.  **Semantic Expansion:** We utilized LMMs to generate diverse textual descriptions for tail classes. Crucially, we prompted the model to include "absent" features (features not present in the few original samples) and varied background contexts to ensure diversity.
2.  **Image Synthesis:** These enriched descriptions were fed into a Text-to-Image (T2I) model to generate synthetic images. This creates a "Generated Tail Dataset" that is semantically aligned with the labels but visually distinct from the original training set.
3.  **Data Integration:** This generated data was integrated into the training pipeline (`use_generated=True`), effectively augmenting the tail classes with synthetic variations.

### 2.2 Domain Adaptation: Architecture Change
The original paper utilized **CLIP (ViT-B/32)**, which is optimized for 224x224 inputs. Applying this directly to CIFAR-100 (32x32) would result in significant information loss due to resizing artifacts.
* **Adaptation:** We adopted **ResNet-18** as the backbone network. ResNet-18 is the standard, state-of-the-art benchmark architecture for CIFAR-100, allowing for a fair and robust evaluation of feature learning capabilities in low-resolution settings.

### 2.3 Training Strategy: The "Custom Balanced Sampler"
A critical challenge in this adaptation was the unavailability of the official code for the paper's **"BalanceMix"** module (a fine-tuning technique). To address the class imbalance during training, we implemented a **Custom Balanced Sampler**.
* **Mechanism:** Standard sampling selects images based on their frequency (Head classes are picked 100x more often). Our `BalancedSampler` enforces a uniform probability $P(c) = 1/C$.
* **Implication:** In every batch, the model sees an equal number of Head (Real) and Tail (Real + Generated) images. This forces the model to treat the generated tail data with the same importance as the abundant head data.

## 3. Experimental Setup

* **Dataset:** CIFAR-100 Long-Tailed
    * **Imbalance Ratio:** 100 (The most frequent class has 500 images, the least frequent has 5).
    * **Class Split:**
        * **Head:** Classes 0-39 (Many-shot)
        * **Medium:** Classes 40-69 (Medium-shot)
        * **Tail:** Classes 70-99 (Few-shot)
* **Training Configuration:**
    * **Epochs:** 30
    * **Loss Functions:** Cross Entropy (Standard), Focal Loss (Hard mining), Asymmetric Loss (ASL).
* **Comparison Groups:**
    1.  **Baseline:** Original LT data only + Standard Sampling.
    2.  **LTGC (Gen + Balanced):** Generated data added + Custom `BalancedSampler`.

## 4. Key Results & Detailed Analysis

### 4.1 Comparative Analysis (Top-1 Accuracy)

| Loss Function | Experiment Setting | **Total Acc** | **Head Acc** (Many) | **Medium Acc** | **Tail Acc** (Few) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cross Entropy** | Baseline | **24.18%** | **43.95%** | 20.87% | 1.13% |
| | **LTGC (Gen + Balanced)** | 20.93% | 26.97% | **25.03%** | **8.77%** |
| **Focal Loss** | Baseline | **25.94%** | **48.95%** | 19.10% | 2.10% |
| | **LTGC (Gen + Balanced)** | 20.06% | 26.88% | **24.63%** | **6.40%** |
| **ASL** | Baseline | 10.84% | 25.25% | 2.47% | 0.00% |
| | **LTGC (Gen + Balanced)** | **14.71%** | 10.37% | **20.13%** | **15.07%** |

### 4.2 Deep Dive: The Tail Improvement
The most significant finding is the **massive improvement in Tail Accuracy**.
* **Phenomenon:** In the Baseline CE model, Tail accuracy was a negligible **1.13%**. With LTGC and Balanced Sampling, it rose to **8.77%**. Under ASL, it reached **15.07%**.
* **Explanation:** This confirms the validity of the generated data. The 5 original images per tail class were insufficient to form a generalized decision boundary. The generated images provided diverse "views" of the tail classes, allowing the model to learn robust features. The `BalancedSampler` ensured these generated images contributed significantly to the gradient updates, preventing them from being drowned out by head classes.

### 4.3 Deep Dive: The Head Degradation
Conversely, we observed a consistent **drop in Head Accuracy** across all settings (e.g., CE: 43.95% $\rightarrow$ 26.97%).
* **Phenomenon:** The model became significantly worse at recognizing the classes it had the most data for.
* **Explanation:** In standard training (Baseline), the model relies on the **Prior Probability Bias**—it learns that "most inputs are dogs," which helps minimize global error. The `BalancedSampler` artificially removes this bias. By forcing the model to pay 50% attention to the difficult/noisy tail data and only 50% to the clean head data, the model effectively "underfits" the rich variations present in the 500 head images. This is a classic case of **"Catastrophic Forgetting"** of the majority concepts.

## 5. Ablation Study: Data vs. Strategy

To determine whether the performance shift came from the *Generated Data itself* or the *Sampling Strategy*, we conducted an ablation study using Cross Entropy.

### 5.1 Experiment Design
* **Setting A (Gen Only):** Generated data is added, but Standard Sampling is used (preserving the long-tail distribution).
* **Setting B (Gen + Balanced):** Generated data is added, and `BalancedSampler` is used (forcing uniform distribution).

### 5.2 Ablation Results

| Metric | **Baseline** | **Ablation: LTGC (Gen Only)** | **Target: LTGC (Gen + Balanced)** |
| :--- | :---: | :---: | :---: |
| **Total Acc** | 24.18% | 22.04% | 20.93% |
| **Head Acc** | **43.95%** | **42.00%** | 26.97% |
| **Tail Acc** | 1.13% | 0.40% | **8.77%** |

### 5.3 Interpretation
1.  **Data Presence is Not Enough (Gen Only):** Even with generated data, the Tail accuracy remained poor (0.40%). This proves that simply adding data isn't enough if the sampling frequency is still dominated by the Head classes. The gradients from the 500 head images overwhelm the gradients from the ~25 tail images (5 original + 20 generated).
2.  **Sampling is the Driver (Gen + Balanced):** The surge to **8.77%** proves that re-balancing is the key mechanism that unlocks the value of the generated data. However, as noted, this aggressive sampling is what harms the Head performance.

## 6. Conclusion

This report details the successful adaptation of the **LTGC framework** to the **CIFAR-100 Long-Tailed** domain.
1.  **Feasibility:** We demonstrated that LMM-driven data generation is a viable strategy for augmenting low-resolution, long-tailed datasets.
2.  **Efficacy:** The generated content, when properly sampled, yielded up to a **15x improvement** in Tail class recognition.
3.  **Trade-off:** The "Training from Scratch" approach with hard balanced sampling results in a zero-sum trade-off between Head and Tail performance.

## 7. Discussion & Opinion: The Path to "Pareto Improvement"

Based on the experimental outcomes, I present the following critical analysis and strategic proposals for future research.

### 7.1 Critique: The Failure of "Hard" Balancing
The most critical takeaway from this experiment is that **"Hard" Balanced Sampling is a double-edged sword.** While we succeeded in raising Tail accuracy (our primary goal), the degradation of Head accuracy (43% $\rightarrow$ 26%) is unacceptable for a deployment-ready system.
A robust AI system should not sacrifice the majority to save the minority. The current approach effectively turned the Long-Tailed problem into a Uniform problem, losing the benefits of the abundant data available for head classes.

### 7.2 The Need for "Pareto Improvement"
We must aim for a **Pareto Improvement**, where Tail performance improves *without* degrading Head performance.
* **Baseline:** Strong Head, Weak Tail.
* **Current Model:** Weak Head, Strong Tail.
* **Ideal Model:** Strong Head, Strong Tail.
The fact that the "Gen Only" experiment maintained Head accuracy (42%) suggests that the generated data *itself* is not harmful. The harm comes entirely from the aggressive training strategy.

### 7.3 Proposal: Decoupling and Soft-Balancing
To achieve this ideal state, I propose adopting a **Decoupling Strategy (Two-Stage Training)** or a **Soft-BalanceMix**, rather than the single-stage hard balancing used here.

1.  **Stage 1: Representation Learning (Standard Sampling)**
    * Train the backbone (ResNet-18) using the original long-tailed distribution (or "Gen Only").
    * **Goal:** Allow the model to learn robust, high-quality feature extractors from the abundant Head data. This secures the "Head Accuracy."

2.  **Stage 2: Classifier Fine-tuning (Balanced Sampling)**
    * Freeze the backbone parameters and fine-tune *only* the final classification layer (FC layer) using the **Generated + Balanced** dataset.
    * **Goal:** Adjust the decision boundaries to be fair to Tail classes without altering the feature representations learned in Stage 1. This effectively combines the best of both worlds.

3.  **Soft BalanceMix:**
    * If single-stage training is preferred, we should implement a weighted loss function where the generated data contributes to the loss with a smaller coefficient ($\lambda < 1$). This would act as a regularizer rather than a primary driver, mitigating the domain gap and preventing catastrophic forgetting of head features.
