# 🛒 Product Category Classification – Machine Learning Project

This project builds a machine learning model that predicts a product category based only on the product title.  
The model is trained using **Support Vector Machine (SVM)** with TF-IDF features and engineered title-based features.

---

## 📁 Project Structure

├── data/

│ └── products.csv

├── notebooks/

│ └── product_title_classification.ipynb    # EDA and preprocessing

├── src/

│ └── predict_category.py   # Script for testing saved model

│ └──  train_model.py   # Script for training and saving the model

└── README.md

---

## ✅ Completed Steps in This Project

### 1. Project Setup
- Created GitHub repository
- Defined folder structure
- Uploaded source dataset

### 2. Exploratory Data Analysis (EDA)
- Loaded dataset from GitHub
- Checked shape, data types, missing values
- Visualized missing values with seaborn heatmap
- Analyzed product category distribution

### 3. Data Cleaning
- Standardized column names
- Removed rows with missing essential fields
- Cleaned category labels (merged duplicates such as “fridge” → “Fridges”)

### 4. Feature Engineering
Created additional features from *product_title*:
- `title_char_count`
- `title_word_count`
- `title_has_numbers`
- `title_has_special_chars`
- `title_uppercase_count`
- `title_longest_word_length`

### 5. Model Training & Evaluation
Tested multiple machine learning models:
- Logistic Regression  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- Support Vector Machine (SVM)

Used:
- `TfidfVectorizer` for text
- MinMaxScaler for numerical features
- `Pipeline` + `ColumnTransformer`

Evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-score  

### 6. Final Model Training (100% Data)
- Trained final model on complete dataset (no train/test split)
- Saved trained pipeline to:
    - model/product_category_svm.pkl

### 7. Inference Script
A fully interactive console script (`predict_category.py`) allows real-time predictions.

---

## 🚀 How to Use

### 📥 Clone Repository
```bash
git clone https://github.com/enis-gosic/product-title-classification-ML-project.git
cd product-title-classification-ML-project
```

### 🔧 Train the Final Model
```bash
cd src
python train_model.py
```

### This generates:
- model/product_category_svm.pkl

### 🔍 Run Category Prediction
```bash
python predict_category.py
```

The script allows you to enter product titles and instantly get predicted categories.

## 👨‍💻 Author
Enis Gosic

This project was developed as part of a practical machine learning module focused on NLP and product classification.

## 📄 License
This project is open-source and free to use.
