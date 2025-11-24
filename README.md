🏡 House Price Prediction — Ames Housing Dataset

Machine Learning Regression Project | Random Forest | Feature Engineering | EDA

This project predicts house prices using the Ames Housing Dataset, a modern and richer alternative to the Boston Housing dataset.
It includes complete EDA, data cleaning, feature engineering, model training, evaluation, and interpretability.

🚀 Project Highlights
✔ End-to-end Machine Learning Pipeline

Data Cleaning

Handling Missing Values

Feature Engineering

Ordinal Encoding of Quality Variables

One-Hot Encoding for Categorical Variables

Train-Test Split

Preprocessing Pipeline (ColumnTransformer)

Regression Modeling & Evaluation

✔ Models Trained
Model	RMSE
Linear Regression	32,923
Ridge Regression	32,955
Lasso Regression	92,955
⭐ Random Forest (Best)	⭐ 27,510
✔ Best Model

Random Forest Regressor

RMSE: 27,510

Strong non-linear performance

Handles categorical + continuous features well

Captures important interactions

📊 Dataset

AmesHousing.csv
(80+ features including numerical, categorical, and quality/condition labels)

Target Variable:

SalePrice → log-transformed (log_saleprice) for stability.

🧹 Data Preprocessing & Cleaning
🔹 Handling Missing Values

Numerical: median imputation

Categorical: “unknown”

LotFrontage: imputed by neighborhood median

GarageYrBlt: imputed from YearBuilt

🔹 Ordinal Encoding (Critical Step)

Quality-related columns (Ex, Gd, TA, Fa, Po):

Converted to numerical scale:

Category	Score
Ex	5
Gd	4
TA	3
Fa	2
Po	1
Unknown	0

Columns encoded:

exter_qual, exter_cond, bsmt_qual, bsmt_cond,
kitchen_qual, garage_qual, garage_cond, heating_qc

🔹 Feature Engineering

New engineered features:

total_sf → total square footage

total_bath → combined full + half + basement baths

house_age → yr_sold – year_built

years_since_remod

total_porch_sf

has_garage, has_basement, has_pool

🧠 Modeling Pipeline

Used Scikit-Learn Pipeline + ColumnTransformer:

Scaling numeric features with StandardScaler

One-Hot encoding categorical features

Combining both for training models

Models tested:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor (Best)

🔝 Top 20 Most Important Features (Random Forest)
1. total_sf              : 0.1275
2. overall_qual          : 0.1169
3. gr_liv_area           : 0.0770
4. house_age             : 0.0631
5. year_built            : 0.0596
6. total_bath            : 0.0532
7. total_bsmt_sf         : 0.0485
8. garage_cars           : 0.0458
9. garage_area           : 0.0437
10. garage_yr_blt        : 0.0426
11. 1st_flr_sf           : 0.0396
12. exter_qual_ta        : 0.0327
13. years_since_remod    : 0.0277
14. year_remod/add       : 0.0269
15. fireplaces           : 0.0245
16. kitchen_qual_ta      : 0.0169
17. totrms_abvgrd        : 0.0156
18. total_porch_sf       : 0.0136
19. exter_qual_gd        : 0.0131
20. lot_frontage         : 0.0125

🧠 Insight:

The top predictors are quality, size, age, and bathrooms.

📈 Evaluation Metric: RMSE

RMSE calculated on original SalePrice by reversing the log transformation.

pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
actual = np.expm1(y_test)
rmse = np.sqrt(mean_squared_error(actual, pred))

📦 Project Structure
House Price Prediction/
│
├── Data/
│   └── AmesHousing.csv
│
├── Notebook/
│   └── house_price_prediction.ipynb
│
├── models/
│   └── house_price_rf_model.pkl
│
├── README.md
│
└── requirements.txt

🔧 How To Run
pip install -r requirements.txt
jupyter notebook


Open the notebook and run cells from top to bottom.

🎯 Final Summary

This project demonstrates:

Strong EDA

Professional feature engineering

Robust preprocessing pipelines

Handling of complex categorical + ordinal features

Multiple regression model comparison

Random Forest–based final prediction

Clear, interpretable feature importance