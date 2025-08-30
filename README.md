# OIBSIP_DataScience_Task3

# 🚗 Car Price Prediction Web App

A **Streamlit-based machine learning web application** that predicts the **selling price of a car**.  
It leverages regression models and interactive data visualizations to help users estimate car prices.  

---

## 📌 Project Overview
Car prices depend on multiple factors such as year of manufacture, kilometers driven, fuel type, and transmission.  
This project provides an **end-to-end ML pipeline** with:  
- Data preprocessing  
- Model training & evaluation  
- Feature importance visualization  
- Interactive prediction interface  

---

## 🔹 Features
- 📂 Upload your dataset in CSV format  
- 📊 Data exploration with visualizations:
  - Price Distribution
  - Year vs Price
  - Fuel Type Analysis
  - Transmission Analysis
  - Correlation Heatmap
- 🤖 Train multiple regression models:
  - Random Forest
  - Linear Regression
  - Decision Tree
  - Gradient Boosting
- 📈 Model evaluation with **MSE** and **R² Score**  
- 🖥️ Predict car price based on user inputs in real time  

---

## 🛠️ Tech Stack
- **Python**  
- **Streamlit** for web app  
- **Scikit-learn** for ML models  
- **Pandas & NumPy** for data handling  
- **Matplotlib & Seaborn** for visualizations  

---

## 📂 Project Structure
```
car_price_prediction.py   # Main Streamlit application
README_Car_Price.md       # Documentation file
```

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone <your_repo_url>
   cd <your_repo_name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run car_price_prediction.py
   ```

4. Upload your dataset or use the provided sample.

---

## 📊 Sample Dataset Format
| Car_Name | Year | Selling_Price | Present_Price | Driven_kms | Fuel_Type | Selling_type | Transmission | Owner |
|----------|------|---------------|---------------|------------|-----------|--------------|--------------|-------|
| car1     | 2018 | 8.5           | 10.5          | 5000       | Petrol    | Dealer       | Manual       | 0     |

---

## 🎯 Demo
[Demo Video on LinkedIn](https://www.linkedin.com/posts/k-duraimurugan-4b83b2307_oasisinfobyte-internship-machinelearning-activity-7367472743044296704-I4fl?utm_source=share&utm_medium=member_desktop&rcm=ACoAAE4w30wBf7N4SVn0jOy8x7aXPJZdKpuYXAs)

---

## 📌 Future Improvements
- Deploy app on Streamlit Cloud or Heroku  
- Add more ML models (XGBoost, CatBoost)  
- Enhance UI with Plotly visualizations  

---
