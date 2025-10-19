# Computational Linear Algebra: PCA & k-Means Clustering

## Project Overview

This project applies **Principal Component Analysis (PCA)** for dimensionality reduction and **k-Means Clustering** for market segmentation on a dataset containing psychological and demographic features.

The core objective is to analyze the underlying data structure, compare the impact of **StandardScaler** and **MinMaxScaler** preprocessing methods on the results, and identify meaningful clusters (segments) of individuals based on their personality traits, interests, and lifestyle features.

---

## Team

- **MOHAMMADI, Arian** (ID: 346278)  
- **KOSE, Mustafa Kerem** (ID: 339018)  

**Academic Year:** 2024/2025

---

## Data & Preprocessing

### Feature Selection

The analysis utilized features from the following categories:

- **Fixed:** Personality, Health  
- **Personal:** Finance  
- **Entertainment:** Movies, Music  

---

### Data Preparation Steps

1. **Sampling:** A random sample of 2/3 of the total rows was used.  
2. **Imputation:**  
   - Missing values were filled using the **mode** for categorical features.  
   - Missing values were filled using the **mean** for numerical features.  
3. **Encoding:** Categorical variables were converted using **Label Encoding**.  
4. **Scaling:** Two datasets were prepared for comparative analysis:  
   - `Xstd_df`: Scaled using **StandardScaler** (Mean ≈ 0, Variance ≈ 1).  
   - `Xmm_df`: Scaled using **MinMaxScaler** (Range [0, 1]).

---

## Key Findings

### PCA for Dimensionality Reduction

PCA was performed to retain components explaining at least **33% of the variance**, capping the maximum number of components at 5.

#### Standard Scaled (`Xstd_df`)
- **Components Used (m):** 5  
- **Rationale:** Chosen for optimal contribution, as equal variance makes the dataset highly suitable for PCA.

#### MinMax Scaled (`Xmm_df`)
- **Components Used (m):** 4  
- **Rationale:** Achieves the target 33% variance with fewer components.

---

### k-Means Clustering Results

The optimal number of clusters (**k = 4**) was determined using the **Silhouette Score** for both datasets.

| Scaling Method | k | Silhouette Score |
|----------------|---|------------------|
| MinMax Scaled  | 4 | **0.161** |
| Standard Scaled | 4 | 0.158 |

> **Observation:** MinMax scaling produced slightly better separation.

---

## Cluster Interpretations (Centroid Analysis)

The four clusters were profiled by interpreting the **Principal Components (PCs)** that defined their cluster centroids:

### 1. High-Energy Social Adventurers
- **Dominant Features:**  
  - High values in *Social Dynamism* (MinMax)  
  - High *Energy Lifestyle* (Standard)  
- **Evaluation Notes:**  
  Strongly validated by *External Evaluation (Age)* — peak representation among younger individuals (19–21).

---

### 2. Emotional & Romantic
- **Dominant Features:**  
  - High *Emotional Instability* (MinMax)  
  - High *Emotional Reactivity* (Standard)  
- **Evaluation Notes:**  
  Linked to emotional/sentimental characteristics, consistent across both scaling methods.

---

### 3. Practical / Financial Focus
- **Dominant Features:**  
  - High *Consumerism/Materialism* (MinMax)  
  - High *Strategic Thinking* (Standard)  
- **Evaluation Notes:**  
  Describes individuals driven by financial concerns or a practical, disciplined approach to life.

---

### 4. Values & Well-being
- **Dominant Features:**  
  - High *Cultural Action/Tradition* (MinMax)  
  - High *Values and Well-being* (Standard)  
- **Evaluation Notes:**  
  Individuals focused on personal growth, empathy, and cultural values.

---

## Dependencies

This analysis requires standard Python libraries for data manipulation and machine learning:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
