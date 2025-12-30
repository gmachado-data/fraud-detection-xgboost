# Fraud Detection with Machine Learning (XGBoost)

## Business Context

This project simulates a real-world scenario of a fintech / payment institution focused on fraud detection.

From a business perspective, fraudulent transactions generate:
- Direct financial losses
- Operational risk
- Liquidity and funding impact
- Customer experience degradation

The objective of this project is to build a machine learning model capable of identifying transactions with a high risk of fraud, supporting preventive actions and risk-based decision making.

---

## Objective

Develop an end-to-end machine learning pipeline to classify fraudulent transactions, prioritizing risk detection and minimizing false negatives, which represent undetected fraud events.

---

## Dataset

This project uses a **public Kaggle dataset** for fraud detection in financial transactions.

**Dataset source:**  
https://www.kaggle.com/datasets/ntnu-testimon/paysim1

To keep the repository lightweight and aligned with good data management practices, the dataset is **not included** in the repository.

### How to run locally

1. Download the dataset from Kaggle.
2. Create a folder named `data/` in the project root.
3. Place the dataset file inside the folder and rename it to:

`transactions.csv`


Expected structure:

```
fraud-detection-xgboost/
│
├── data/
│ └── transactions.csv (not versioned)
├── fraud_detection_xgboost.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Approach

The pipeline follows these steps:

1. **Risk Problem Definition**
   - Fraud as a financial and operational risk
   - Cost-sensitive classification problem

2. **Exploratory Data Analysis (EDA)**
   - Transaction behavior analysis
   - Amount distribution and outlier detection
   - Temporal patterns and transaction volume
   - Category-level transaction concentration

3. **Data Preparation**
   - Encoding categorical variables
   - Removal of identifiers and non-predictive attributes
   - Bias and overfitting risk mitigation

4. **Model Benchmark**
   - Logistic Regression
   - Random Forest
   - KNN
   - XGBoost
   - LightGBM

5. **Model Selection**
   - Focus on Recall and AUC due to class imbalance
   - XGBoost selected for performance, scalability, and flexibility

6. **Threshold Optimization**
   - Threshold treated as a business decision
   - Trade-off between fraud detection (recall) and operational cost (false positives)

---

## Metrics and Business Rationale

Fraud detection is a highly imbalanced problem (~1.2% fraud rate).  
Therefore, **accuracy is not a reliable metric**.

The project prioritizes:
- **Recall** → reduce undetected fraud (false negatives)
- **AUC** → overall ranking ability of the model
- **F1-score** → balance between precision and recall

Threshold tuning is used to align the model behavior with business risk tolerance.

---

## Results

- XGBoost achieved strong performance across AUC and Recall
- Threshold adjustment significantly improved fraud detection
- Final threshold favors recall, reducing financial exposure to fraud at the cost of manageable operational overhead

---

## How to Run

```bash
pip install -r requirements.txt
python fraud_detection_xgboost.py
```
---

## Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---

## Notes

This project focuses on business-oriented decision making for fraud detection, emphasizing cost-sensitive classification and risk mitigation rather than pure metric optimization.