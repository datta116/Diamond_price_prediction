# 💎 Diamond Price Prediction

A **machine learning model** to predict diamond prices (in INR) using physical characteristics like carat, cut, color, clarity, and dimensions.

> **Dataset:** [Diamonds Prices 2022](https://www.kaggle.com/datasets/shivam2503/diamonds) (53,943 diamonds)  
> **Model:** Random Forest Regressor  
> **Target:** `Price_in_rs` (USD × 82.88)  
> **Status:** Completed & Deployable

---

## Output Overview
**Live Prediction Example**  
![Diamond Price Prediction App](https://github.com/datta116/Diamond_price_prediction/blob/52fc1c25dbe42ffd0ba754ac898378c747301cd5/Docs/Screenshot%202025-11-16%20073556.png)

## 📁 Project Structure

.
├── Diamondprice_prediction.ipynb     # Main notebook
├── DiamondsPrices2022.csv            # Dataset
├── diamond_price_model.pkl           # Trained model
├── README.md                         # This file
└── requirements.txt                  # (Optional) pip freeze > requirements.txt

## 📊 Project Overview

This project performs **end-to-end data analysis and modeling** to predict diamond prices with high accuracy using the classic **Diamonds dataset**.

### Key Features:
- Data cleaning & exploration
- Feature engineering (square root transformation for skewness)
- Categorical encoding (Ordinal: Cut, Color, Clarity)
- Currency conversion: USD → INR
- Model training with **Random Forest**
- Model saved using `pickle` for reuse

---

## 🛠 Tech Stack

| Component           | Tool/Library                     |
|---------------------|----------------------------------|
| Language            | Python                           |
| Data Processing     | `pandas`, `numpy`                |
| Visualization       | `matplotlib`, `seaborn`          |
| Machine Learning    | `scikit-learn`                   |
| Model Persistence   | `pickle`                         |

---

## 📈 Data Preprocessing

1. **Loaded dataset** → `DiamondsPrices2022.csv`
2. **No missing values or duplicates**
3. **Numerical Features:**
   - Applied **square root transformation** on `carat` and `depth` to reduce skewness
4. **Categorical Features:**
   - Ordinal encoding:
     - `cut`: Fair=0, Good=1, Very Good=2, Premium=3, Ideal=4
     - `color`: J=0, I=1, ..., D=6
     - `clarity`: I1=0, SI2=1, ..., IF=7
5. **Target Variable:**
   - Created `Price_in_rs = price * 82.88` (USD to INR conversion)

---

## 🤖 Model Training

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

X = df_new[['carat', 'depth', 'table', 'clarity', 'color', 'cut']]
y = df_new['Price_in_rs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
with open('diamond_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```
## 🚀 Final Prediction
```Python
# Load the model from the pickle file
with open('diamond_price_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Use the loaded model to make predictions
new_data = df_new[['carat', 'depth', 'table', 'clarity', 'color', 'cut']].tail(1)
prediction = loaded_model.predict(new_data)
print("The model predicts the price for the last row:", prediction[0])
print("Actual value is:", df_new['Price_in_rs'].iloc[-1])  # Fetching the actual value from the dataframe
```



