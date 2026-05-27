# 🤖 Machine Learning Fundamentals — End-to-End Implementations

> A structured collection of end-to-end machine learning implementations covering supervised learning, support vector machines, unsupervised learning, and deep learning with hyperparameter tuning. Built as part of MS AI & Automation studies at University West, Sweden.

---

## 📚 Notebooks

### 1. 🏠 California Housing Price Prediction
**File:** `california_housing_regression.ipynb`

End-to-end regression pipeline predicting median house values across Californian districts using the classic California Housing dataset (20,640 instances, 10 features).

**What's covered:**
- Stratified train/test splitting to preserve income category distribution
- Custom feature engineering (`rooms_per_household`, `bedrooms_per_room`, `population_per_household`)
- Full preprocessing pipeline with `SimpleImputer`, `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`
- Custom `BaseEstimator` / `TransformerMixin` transformer
- Model training and comparison: Linear Regression, Decision Tree, Random Forest, SVR
- Hyperparameter tuning with both `GridSearchCV` and `RandomizedSearchCV`
- Feature importance analysis for Random Forest and Decision Tree
- 95% confidence interval on final test set RMSE

**Key results:**

| Model | CV RMSE (Mean) |
|---|---|
| Linear Regression | 69,104 |
| Decision Tree | 71,630 |
| Random Forest *(best)* | **49,118** |
| SVR | 83,058 |

**Tech:** `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

### 2. 🔲 Support Vector Machines
**File:** `support_vector_machines.ipynb`

Deep dive into SVM theory and application covering classification, regression, and kernel methods. Includes a from-scratch linear SVM implementation using batch gradient descent.

**What's covered:**
- Linear SVM classification with hard and soft margin
- Feature scaling sensitivity demonstration
- Nonlinear classification with polynomial features and kernel trick
- Polynomial kernel, RBF kernel — hyperparameter effects visualised
- SVM regression with epsilon-insensitive tube (LinearSVR, SVR)
- Custom `MyLinearSVC` implementation from scratch using gradient descent
- Multi-kernel comparison on MNIST (70,000 images, 10 classes)
- Comparison against KNN, SGD, and Random Forest classifiers

**MNIST Classification Results:**

| Model | Accuracy | Training Time |
|---|---|---|
| Linear SVM | 93.3% | 14,911s |
| Polynomial SVM *(best)* | **97.6%** | 10,782s |
| RBF SVM | 97.0% | 16,074s |
| KNN | 94.9% | 3s |
| Random Forest | 96.5% | 94s |

**Tech:** `scikit-learn` · `numpy` · `matplotlib`

---

### 3. 🔵 Unsupervised Learning & Clustering
**File:** `unsupervised_learning_clustering.ipynb`

Comprehensive exploration of unsupervised learning techniques with real-world applications including image segmentation, semi-supervised learning, and anomaly detection.

**What's covered:**
- K-Means: algorithm internals, variability, inertia, elbow method
- Silhouette analysis for optimal cluster selection
- Mini-Batch K-Means vs standard K-Means (speed/accuracy tradeoff)
- K-Means for image segmentation (ladybug image, 2–10 colour reduction)
- K-Means as preprocessing for classification (35% error reduction on digits dataset)
- Semi-supervised learning: label propagation from 50 labelled instances → 92.7% accuracy
- DBSCAN: density-based clustering with anomaly detection
- Spectral Clustering, Agglomerative Clustering
- Gaussian Mixture Models: EM algorithm, BIC/AIC model selection, anomaly detection
- Bayesian GMM with automatic component selection
- Olivetti Faces dataset: PCA + K-Means clustering (120 clusters, silhouette-optimised)
- Reconstruction error for anomaly detection using PCA

**Tech:** `scikit-learn` · `scipy` · `numpy` · `matplotlib` · `seaborn`

---

### 4. 🧠 MLP Hyperparameter Tuning on MNIST
**File:** `mlp_hyperparameter_tuning_mnist.ipynb`

Automated neural architecture search using Keras Tuner to find the optimal MLP configuration for MNIST digit classification. Compares tuned model against a baseline.

**What's covered:**
- MNIST data loading, normalisation, and flattening
- Baseline MLP (2 hidden layers × 128 units) as reference
- `RandomSearch` tuner over: number of layers (2–5), units per layer (64–512), learning rate, batch size
- 10 trials × 50 epochs each (2h 24m total search time on GPU)
- Best architecture: 4 layers (448 → 384 → 448 → 256), lr=0.0001, batch=96
- Training/validation accuracy and loss curves for best model
- Full hyperparameter search summary with trial-by-trial scores

**Results:**

| Model | Test Accuracy |
|---|---|
| Baseline MLP | 97.75% |
| Tuned MLP *(Keras Tuner)* | **97.72%** |

**Tech:** `TensorFlow` · `Keras` · `Keras Tuner` · `numpy` · `matplotlib`

---

## 🗂️ Repository Structure

```
EndtoEndMachineLearningLab/
├── california_housing_regression.ipynb    # Regression pipeline + model comparison
├── support_vector_machines.ipynb          # SVM theory + MNIST classification
├── unsupervised_learning_clustering.ipynb # Clustering + GMM + anomaly detection
├── mlp_hyperparameter_tuning_mnist.ipynb  # Neural network + Keras Tuner
└── README.md
```

---

## 🚀 Getting Started

### Clone the repository
```bash
git clone https://github.com/BinuShefieldShifani/EndtoEndMachineLearningLab.git
cd EndtoEndMachineLearningLab
```

### Run in Google Colab (Recommended)
Each notebook is self-contained and runs on Google Colab with free GPU access. Click the **Open in Colab** badge or go to [colab.research.google.com](https://colab.research.google.com), open via GitHub, and select the notebook.

### Run locally
```bash
pip install scikit-learn pandas numpy matplotlib seaborn tensorflow keras-tuner scipy
jupyter notebook
```

---

## 📦 Dependencies

```
scikit-learn>=0.20
pandas
numpy
matplotlib
seaborn
tensorflow
keras-tuner
scipy
jupyter
```

---

## 👤 Author

**Binu Shefield Shifani**

Software Engineer (5 years, Cognizant Technology Solutions)
MS AI & Automation · University West, Trollhättan, Sweden
[GitHub](https://github.com/BinuShefieldShifani)
