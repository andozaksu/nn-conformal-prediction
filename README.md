# 🤖 Nearest-Neighbour Conformal Prediction

A 1-Nearest Neighbour classifier and a conformal predictor, both implemented from scratch in NumPy and evaluated on the Iris and Ionosphere datasets.

📓 [View the notebook](nn_conformal_prediction.ipynb)

## 📌 Overview

Conformal prediction wraps a classifier so that, instead of returning a single label, it returns a **p-value for every possible class**. A low p-value for a class means the new example looks unusual for that class. This project builds the whole pipeline by hand: the distance calculations, the 1-NN classifier, the conformity scores and the p-values.

## ⚙️ How It Works

1. **1-NN classifier:** pairwise Euclidean distances are computed with a vectorised NumPy matrix, and each test point takes the label of its nearest training point.
2. **Conformity score:** for each example, the distance to the nearest example of a *different* class divided by the distance to the nearest example of the *same* class. A high score means the example sits comfortably inside its class.
3. **Nonconformity:** the inverse of the conformity score.
4. **p-values:** for each test point and each candidate label, the point is added to the training set with that label, all scores are recomputed, and the p-value is the proportion of examples at least as nonconforming as the new one.
5. **Average false p-value:** the mean p-value assigned to the *wrong* classes. Lower is better, because it means the predictor rarely finds incorrect labels plausible.

## 📊 Results

Stratified 75/25 train-test split, with a fixed random seed for reproducibility.

| Dataset | 1-NN test error | Average false p-value |
|---------|-----------------|-----------------------|
| Iris | 7.9% | 0.011 |
| Ionosphere | 18.2% | 0.065 |

## 📂 Data

* **Iris:** loaded through scikit-learn
* **Ionosphere:** included as `ionosphere.txt`, from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/52/ionosphere)

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** NumPy (all the algorithms), scikit-learn (loading Iris and splitting the data)

## 🚀 Running the Project

1. Clone the repo:
```
   git clone https://github.com/andozaksu/nn-conformal-pr