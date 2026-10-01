# ✈️ Flight Price Prediction

A machine learning regression project that predicts flight ticket prices based on flight-related features such as airline, journey date, source, destination, departure and arrival time, duration, and number of stops.

The project covers data cleaning, feature engineering, categorical encoding, exploratory analysis, regression modeling, model comparison, hyperparameter tuning, feature importance analysis, and prediction error analysis using Python and Scikit-learn.

---

## 📌 Project Overview

Flight ticket prices can vary significantly depending on factors such as airline, travel date, route, journey duration, departure time, arrival time, and number of stops.

The objective of this project is to build a regression model capable of estimating flight prices from these characteristics.

The project experiments with:

* Random Forest Regression
* Decision Tree Regression
* Random Forest hyperparameter tuning using GridSearchCV
* Model evaluation using MAE, MSE, RMSE, and R²
* Feature importance analysis
* Actual vs. predicted price visualization
* Prediction error analysis

---

## 🎯 Objective

The main objectives of this project are to:

1. Load and inspect flight booking data.
2. Handle missing values.
3. Extract useful features from date and time columns.
4. Convert flight duration into numerical minutes.
5. Convert the number of stops into numerical values.
6. Encode categorical variables using one-hot encoding.
7. Train regression models for flight price prediction.
8. Compare Random Forest and Decision Tree performance.
9. Tune the Random Forest model using GridSearchCV.
10. Analyze feature importance and prediction errors.

---

## 📂 Dataset

The project contains three main datasets/data artifacts.

### `Data_Train.xlsx`

The primary training dataset contains:

* **10,683 rows**
* **11 columns**

Columns:

| Feature           | Description                   |
| ----------------- | ----------------------------- |
| `Airline`         | Airline operating the flight  |
| `Date_of_Journey` | Date of the journey           |
| `Source`          | Starting location             |
| `Destination`     | Destination location          |
| `Route`           | Flight route                  |
| `Dep_Time`        | Departure time                |
| `Arrival_Time`    | Arrival time                  |
| `Duration`        | Flight duration               |
| `Total_Stops`     | Number of stops               |
| `Additional_Info` | Additional flight information |
| `Price`           | Target flight ticket price    |

The dataset initially contains missing values in `Route` and `Total_Stops`. Rows containing missing values are removed during preprocessing, resulting in **10,682 usable records**.

---

### `Test_set.xlsx`

The test dataset contains:

* **2,671 rows**
* **10 columns**

It contains the flight-related input features but does not contain the `Price` target column.

The notebook provided with this project does **not directly use this file for its reported model evaluation**. The reported MAE, RMSE, and R² metrics come from an 80/20 train-test split of the cleaned `Data_Train.xlsx` dataset.

---

### `deploy_df.csv`

`deploy_df.csv` contains **10,682 rows and 29 columns** and represents a processed dataset containing encoded airline/source/destination features along with engineered numerical features and the price column.

It includes features such as:

* Airline one-hot encoded variables
* Source one-hot encoded variables
* Destination one-hot encoded variables
* Arrival time
* Total stops
* Price
* Day and month of journey
* Departure hour and minute
* Duration hour and minute

This file is included as a processed data artifact for downstream/deployment-oriented use.

---

## 🔄 Project Workflow

```text
Raw Flight Data
      │
      ▼
Data Loading
      │
      ▼
Data Inspection
      │
      ▼
Missing Value Handling
      │
      ▼
Date & Time Feature Engineering
      │
      ├── Journey Day
      ├── Journey Month
      ├── Departure Hour
      ├── Departure Minute
      ├── Arrival Hour
      └── Arrival Minute
      │
      ▼
Duration Conversion
      │
      ▼
Total Stops → Numerical Encoding
      │
      ▼
Remove Unused Features
      │
      ├── Route
      └── Additional_Info
      │
      ▼
One-Hot Encoding
      │
      ├── Airline
      ├── Source
      └── Destination
      │
      ▼
Feature / Target Separation
      │
      ▼
80/20 Train-Test Split
      │
      ├───────────────┐
      ▼               ▼
Random Forest     Decision Tree
      │               │
      └───────┬───────┘
              ▼
       Model Evaluation
              │
              ▼
     Random Forest Tuning
        using GridSearchCV
              │
              ▼
     Feature Importance
              │
              ▼
       Error Analysis
```

---

## 🧹 Data Preprocessing

### 1. Missing Values

Missing values are inspected using:

```python
df.isnull().sum()
```

Rows containing missing values are removed using:

```python
df.dropna(inplace=True)
```

This reduces the training dataset from **10,683 to 10,682 rows**.

---

### 2. Journey Date Features

`Date_of_Journey` is converted into separate numerical features:

* `Journey_Day`
* `Journey_Month`

Example:

```python
df["Journey_Day"] = pd.to_datetime(
    df.Date_of_Journey,
    format="%d/%m/%Y"
).dt.day

df["Journey_Month"] = pd.to_datetime(
    df.Date_of_Journey,
    format="%d/%m/%Y"
).dt.month
```

---

### 3. Departure Time Features

Departure time is separated into:

* `Dep_Hour`
* `Dep_Minute`

---

### 4. Arrival Time Features

Arrival time is separated into:

* `Arrival_Hour`
* `Arrival_Minute`

---

### 5. Flight Duration

The original `Duration` feature contains values such as:

```text
2h 50m
7h 25m
4h
```

A custom function converts duration into total minutes.

The resulting feature is:

```text
Duration_Minutes
```

For example:

```text
2h 50m → 170 minutes
```

---

### 6. Number of Stops

The categorical `Total_Stops` feature is converted into numerical values:

| Original Value | Encoded Value |
| -------------- | ------------: |
| `non-stop`     |             0 |
| `1 stop`       |             1 |
| `2 stops`      |             2 |
| `3 stops`      |             3 |
| `4 stops`      |             4 |

---

### 7. Removing Features

The notebook removes:

```text
Additional_Info
Route
```

It also removes the original versions of the features after engineering/encoding:

```text
Airline
Date_of_Journey
Source
Destination
Dep_Time
Arrival_Time
Duration
```

The final modeling dataset contains **28 predictor variables**.

---

## 🔤 Categorical Encoding

One-hot encoding is applied to:

* `Airline`
* `Source`
* `Destination`

The implementation uses:

```python
pd.get_dummies(..., drop_first=True)
```

This converts categorical flight information into numerical variables that can be used by the regression models.

After preprocessing:

```text
X → 10,682 rows × 28 features
```

---

## 🤖 Machine Learning Models

### 1. Random Forest Regressor

The first major model is:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

The dataset is divided using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Result:

```text
Training samples: 8,545
Testing samples: 2,137
```

---

### 2. Decision Tree Regressor

A Decision Tree model is also trained:

```python
DecisionTreeRegressor(
    random_state=42
)
```

Its performance is compared against the Random Forest model.

---

## 📊 Model Evaluation

The following regression metrics are calculated:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R² Score

### Random Forest

| Metric |       Result |
| ------ | -----------: |
| MAE    |     1,157.62 |
| MSE    | 3,850,463.02 |
| RMSE   |     1,962.26 |
| R²     |   **0.8214** |

### Decision Tree

| Metric |       Result |
| ------ | -----------: |
| MAE    |     1,286.80 |
| MSE    | 4,759,350.38 |
| RMSE   |     2,181.59 |
| R²     |       0.7793 |

The Random Forest model in the notebook achieved an R² of approximately **0.8214** on the held-out test split.

---

## ⚙️ Random Forest Hyperparameter Tuning

GridSearchCV is used to explore different Random Forest configurations.

The parameter grid includes:

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
```

Three-fold cross-validation is used with:

```python
scoring="r2"
```

The best cross-validation configuration reported by the notebook is:

```text
n_estimators = 200
max_depth = 20
min_samples_split = 2
min_samples_leaf = 2
```

Best cross-validation R²:

```text
0.7817
```

When evaluated on the held-out test set, the tuned Random Forest produced:

| Metric | Tuned Random Forest |
| ------ | ------------------: |
| MAE    |            1,129.81 |
| MSE    |        3,966,658.33 |
| RMSE   |            1,991.65 |
| R²     |              0.8160 |

The tuned model reduced MAE compared with the initial Random Forest, while its held-out R² was slightly lower than the initial Random Forest result.

---

## 📈 Feature Importance

The Random Forest feature importance analysis identifies the features contributing most strongly to the model's predictions.

The leading features reported in the notebook include:

| Feature                | Importance |
| ---------------------- | ---------: |
| `Duration_Minutes`     |     0.4657 |
| `Journey_Day`          |     0.1280 |
| `Journey_Month`        |     0.0634 |
| `Jet Airways Business` |     0.0628 |
| `Jet Airways`          |     0.0579 |
| `Arrival_Hour`         |     0.0374 |
| `Total_Stops`          |     0.0329 |
| `Dep_Hour`             |     0.0304 |
| `Dep_Minute`           |     0.0265 |
| `Arrival_Minute`       |     0.0221 |

`Duration_Minutes` has the highest feature importance in the trained Random Forest model.

---

## 📉 Prediction Visualization

The notebook visualizes model performance using an **Actual vs Predicted Flight Prices** scatter plot.

The visualization compares:

```text
Actual Price
       vs.
Predicted Price
```

A reference diagonal line is also plotted to help visualize how closely predictions follow actual prices.

---

## 🔎 Prediction Error Analysis

The project also investigates prediction errors.

The error is calculated as:

```python
Error = Actual Price - Predicted Price
```

The notebook calculates:

* Mean error
* Median error
* Minimum error
* Maximum error
* Absolute error

For the initial Random Forest predictions:

```text
Mean Error:    -76.76
Median Error:  -36.34
Minimum Error: -7,435.78
Maximum Error: 33,654.60
```

The notebook also identifies the ten predictions with the largest absolute errors and examines their corresponding flight characteristics.

This provides additional insight into cases where the model has difficulty estimating ticket prices.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* Random Forest Regression
* Decision Tree Regression
* GridSearchCV

### Evaluation

* Mean Absolute Error
* Mean Squared Error
* Root Mean Squared Error
* R² Score

### Visualization

* Matplotlib

### Data Formats

* Excel (`.xlsx`)
* CSV (`.csv`)
* Jupyter Notebook (`.ipynb`)

---

## 📁 Project Structure

```text
Flight-Price-Prediction/
│
├── PRJ-12_Flight_Price_Prediction.ipynb
│
├── Data_Train.xlsx
│
├── Test_set.xlsx
│
├── deploy_df.csv
│
└── README.md
```

> The notebook originally references a local Windows file path for `Data_Train.xlsx`. For GitHub sharing and reproducibility, the notebook should ideally be updated to use a relative path such as `pd.read_excel("Data_Train.xlsx")`.

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Flight-Price-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

```text
PRJ-12_Flight_Price_Prediction.ipynb
```

### 5. Run the cells

Make sure the following files are located in the same project directory:

```text
Data_Train.xlsx
Test_set.xlsx
deploy_df.csv
```

---

## 📌 Key Results

The project demonstrates that tree-based regression models can capture relationships between flight characteristics and ticket prices.

The initial Random Forest model achieved:

```text
R²     : 0.8214
MAE    : 1157.62
RMSE   : 1962.26
```

The model's feature importance analysis identified `Duration_Minutes` as the most important feature among the engineered and encoded predictors used by the model.

---

## 🚀 Potential Improvements

Possible future improvements include:

* Use the provided `Test_set.xlsx` as a separate prediction dataset.
* Build a reusable preprocessing pipeline with Scikit-learn `Pipeline` and `ColumnTransformer`.
* Save the trained model using `joblib` or `pickle`.
* Build a Streamlit interface for interactive flight price prediction.
* Perform additional hyperparameter optimization.
* Compare additional regression algorithms such as Gradient Boosting, XGBoost, or Random Forest variants.
* Improve categorical feature handling.
* Investigate the high-error predictions identified during error analysis.
* Remove hard-coded local file paths and use relative paths for reproducibility.

---

## 📚 Project Type

**Machine Learning | Predictive Analytics | Regression | Flight Price Prediction**

---

## 👨‍💻 Author

**Harshal Shirsat**

B.Tech – Bioengineering Science & Research

Interested in Data Science, Data Analytics, Machine Learning, and Healthcare Analytics.
