# Titanic-disaster

## Titanic ML — Reproducible Docker Setup (Python + R)

This repository contains two **fully containerized** implementations of a Titanic survivorship predictor:

- **Python container**: trains a logistic regression model, prints **train accuracy** and save test result as **predictions.csv**.  
- **R container**: equivalent workflow using base-R `glm(family = binomial)` and prints **accuracy only** and save test result as **predictions.csv**.

---

### 0) Prerequisites

- **Docker** installed and running  
  - Windows: Docker Desktop  
  - macOS: Docker Desktop  
  - Linux: Docker Engine (`docker --version` should work)
- (Optional) **Git** for cloning the repo

If Docker Desktop isn’t running, builds or containers will fail.

---

### 1) Repo Structure

```text
project_root/
├── src/
│   ├── app/             # Python implementation
│   │   └── main.py
│   ├── appR/            # R implementation
│   │   └── main.R
│   └── data/            # Dataset goes here (see Step 2)
│       ├── train.csv
│       ├── test.csv
│       └── gender_submission.csv
├── Dockerfile           # Python container
├── Dockerfile_R         # R container
├── install_packages.R   # (optional) R package installer; base R only by default
├── requirements.txt     # Python dependencies
└── README.md
```
Paths inside both containers are wired so that src/data/ is the working folder for data files.
Do not change filenames.

### 2) Download the Data (Kaggle Titanic)

1. Go to Kaggle’s **Titanic: Machine Learning from Disaster** competition.  
2. Download these two files:
   - `train.csv`
   - `test.csv`
3. Place all three files into **`src/data/`** (create it if it doesn’t exist).

Your data folder should look like:

```text
src/data/
├─ train.csv
└─ test.csv
```

**No unzipping or renaming needed.**  
The code expects these exact filenames.

---

### 3) Quick Start (Everything)

**From the project root:**  
docker build -f Dockerfile     -t titanic_py .  
docker build -f Dockerfile_R   -t titanic_r  .  

docker run --rm -v ${PWD}\src\data:/app/src/data titanic_py 
docker run --rm -v ${PWD}\src\data:/app/src/data titanic_r  
You should see model logs and final accuracy for each container.

### 4) Python Container and Expected Output
#### 4.1 Build
`docker build -f Dockerfile -t titanic_py .`

#### 4.2 Run
`docker run --rm -v ${PWD}\src\data:/app/src/data titanic_py`

**Expected Output (example):**

=== Titanic Logistic Regression (accuracy only) ===
=== Titanic Logistic Regression — Train Accuracy + Predictions ===

[1] Loading datasets from ./data ...
train.shape: (891, 12)
test.shape:  (418, 11)

[2] Preparing data ...

[3] Encoding categorical variables ...
 - Encoding 'Sex'
 - One-hot encoding 'Embarked' (drop_first=True)
 - Dropping columns: ['Name', 'Ticket', 'Cabin']

[4] Imputing NAs in numeric columns with train medians ...

[5] Scaling numeric features using z-score scaling ...

[6] Training Logistic Regression ...
 - Model trained successfully.

[7] Evaluating training accuracy ...
Training Accuracy: shows the training accuracy

[8] Generating predictions on test set ...
Saved predictions to data/predictions.csv

=== Done. ===

### 5) R Container and Expected Output
#### 5.1 Build
`docker build -f Dockerfile_R -t titanic_r .`

#### 5.2 Run
`docker run --rm -v ${PWD}\src\data:/app/src/data titanic_r`

Expected Output:
Mirrors the Python version and ends with Training Accuracy and Test Accuracy.

### 6)What Both Scripts Do
1. Load train.csv and test.csv.
2. Encode Sex (male=0, female=1).
3. One-hot encode Embarked (drop first level).
4. Drop text columns: Name, Ticket, Cabin.
5. Impute missing numeric values using train medians.
6. Apply z-score scaling (mean/sd from train set).
7. Fit logistic regression model
8. Print training accuracy.
9. Save test result as `prediction.csv` to src/data.

### 7)One-Screen Quick Guide

#### 1) Put the Kaggle CSVs in: src/data/
####    - train.csv
####    - test.csv
####    - gender_submission.csv

#### 2) Build both images:  
docker build -f Dockerfile   -t titanic_py .  
docker build -f Dockerfile_R -t titanic_r  .  

# 3) Run both:  
docker run --rm -v ${PWD}\src\data:/app/src/data titanic_py  
 
docker run --rm -v ${PWD}\src\data:/app/src/data titanic_r 

