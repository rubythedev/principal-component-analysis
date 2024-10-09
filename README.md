# PCA_COV: Advanced Principal Component Analysis Using Covariance Matrix

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-yellow.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)

## 📈 Overview

**PCA_COV** is a high-performance Python library designed for performing Principal Component Analysis (PCA) using the covariance matrix approach. Tailored for data analysts, data scientists, and professionals seeking data science roles, this tool simplifies dimensionality reduction, enhances data visualization, and facilitates insightful exploratory data analysis. Whether you're working on machine learning projects, statistical modeling, or data preprocessing, PCA_COV is your go-to solution for transforming complex datasets into actionable insights.

## 🚀 Key Features

- **Comprehensive PCA Implementation:** Execute PCA using the covariance matrix method to uncover underlying data structures.
- **Variance Analysis:** Calculate and visualize both the proportion and cumulative variance explained by each principal component.
- **Data Normalization:** Option to normalize datasets, ensuring optimal performance and accuracy in PCA.
- **Dimensionality Reduction:** Efficiently reduce high-dimensional data to lower dimensions while preserving essential information.
- **Projection Capabilities:** Project data onto selected principal components for streamlined analysis and visualization.
- **Elbow Plot Generation:** Automatically generate elbow plots to determine the optimal number of principal components to retain.
- **Data Reconstruction:** Reconstruct original data from principal components, enabling approximation and noise reduction.

## 🛠️ Technologies & Skills

- **Programming Languages:** Python 3.x
- **Libraries & Frameworks:** 
  - [NumPy](https://numpy.org/) for numerical computations and matrix operations
  - [Pandas](https://pandas.pydata.org/) for data manipulation and preprocessing
  - [Matplotlib](https://matplotlib.org/) for data visualization and plotting
- **Data Science Techniques:** 
  - Dimensionality Reduction
  - Exploratory Data Analysis (EDA)
  - Statistical Modeling
  - Machine Learning Preprocessing
- **Version Control:** Git & GitHub

## 💡 Why PCA using Covariance?

In the realm of data science, understanding and simplifying complex datasets is crucial. Principal Component Analysis with Covariance empowers data analysts and scientists to:

- **Enhance Data Insights:** Reveal hidden patterns and relationships within high-dimensional data.
- **Improve Model Performance:** Reduce dimensionality to eliminate multicollinearity, enhancing the performance of machine learning models.
- **Optimize Computational Efficiency:** Streamline data processing, reducing computational load and speeding up analysis.
- **Facilitate Data Visualization:** Transform data into lower dimensions, making it easier to visualize and interpret.

## 📚 Getting Started

### 🔧 Retrieve Data

1. **Download the Data**
   - Download the zipped folder from [Google Drive](https://drive.google.com/file/d/1p2PZC2RBFu1vQmgUg1hTB20Z8URIMmSl/view?usp=drive_link).

### 📝 Usage

1. **Import the PCA_COV Class**
    ```python
    from pca_cov import PCA_COV
    import pandas as pd
    ```

2. **Load Your Dataset**
    ```python
    data = pd.read_csv('your_dataset.csv')
    ```

3. **Initialize PCA_COV**
    ```python
    pca = PCA_COV(data)
    ```

4. **Perform PCA**
    ```python
    selected_vars = ['feature1', 'feature2', 'feature3']
    pca.pca(vars=selected_vars, normalize=True)
    ```

5. **View Variance Explained**
    ```python
    print("Proportion of Variance:", pca.get_prop_var())
    print("Cumulative Variance:", pca.get_cum_var())
    ```

6. **Generate Elbow Plot**
    ```python
    pca.elbow_plot(num_pcs_to_keep=5)
    plt.show()
    ```

7. **Project Data onto Principal Components**
    ```python
    projected_data = pca.pca_project(pcs_to_keep=[0, 1])
    print(projected_data)
    ```

### 📈 Example Project: Facial Recognition with PCA

This example demonstrates how to apply PCA_COV to a facial recognition dataset, reducing dimensionality and visualizing reconstructed images.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pca_cov
import time

# Load dataset
face_imgs = np.load('data/lfwcrop.npy')
with open('data/lfwcrop_ids.txt') as fp:
    face_names = fp.read().splitlines()

# Prepare data
num_samples = face_imgs.shape[0]
face_data = face_imgs.reshape(num_samples, -1)
face_df = pd.DataFrame(face_data)
face_df.columns = range(face_df.shape[1])

# Initialize PCA
pca_obj = pca_cov.PCA_COV(face_df)

# Perform PCA
start_time = time.time()
pca_obj.pca(list(range(face_df.shape[1])))
end_time = time.time()
print("PCA processing time:", end_time - start_time, "seconds")

# Elbow plot
pca_obj.elbow_plot(num_pcs_to_keep=200)
plt.show()

# Reconstruct images using different numbers of principal components
def make_imgs(reconstructed_imgs):
    img_size = int(np.sqrt(reconstructed_imgs.shape[1]))
    return reconstructed_imgs.reshape(-1, img_size, img_size)

def face_plot(face_imgs, face_names):
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            ax = axs[i, j]
            ax.imshow(face_imgs[idx], cmap='gray')
            ax.set_title(face_names[idx])
            ax.axis('off')
    plt.tight_layout()
    plt.show()

# 50% Variance
num_pcs_50 = pca_obj.variance_accounted_for(0.5)
reconstructed_data_50 = pca_obj.pca_then_project_back(num_pcs_50)
inflated_imgs_50 = make_imgs(reconstructed_data_50)
face_plot(inflated_imgs_50, face_names)

# 70% Variance
num_pcs_70 = pca_obj.variance_accounted_for(0.7)
reconstructed_data_70 = pca_obj.pca_then_project_back(num_pcs_70)
inflated_imgs_70 = make_imgs(reconstructed_data_70)
face_plot(inflated_imgs_70, face_names)

# 80% Variance
num_pcs_80 = pca_obj.variance_accounted_for(0.8)
reconstructed_data_80 = pca_obj.pca_then_project_back(num_pcs_80)
inflated_imgs_80 = make_imgs(reconstructed_data_80)
face_plot(inflated_imgs_80, face_names)

# 95% Variance
num_pcs_95 = pca_obj.variance_accounted_for(0.95)
reconstructed_data_95 = pca_obj.pca_then_project_back(num_pcs_95)
inflated_imgs_95 = make_imgs(reconstructed_data_95)
face_plot(inflated_imgs_95, face_names)
