# 🔋 Household Energy Usage Forecast

## 📌 Project Overview
Household Energy Usage Forecast is a **machine learning-based web application** designed to **predict household energy consumption**. The project helps in analyzing patterns and optimizing energy usage based on various factors like time of the day, weekday, and historical trends.

This application is built using **Streamlit** for the interactive dashboard and **machine learning models** trained to forecast energy consumption.

---

## 🎯 **Key Features**
✔ **Predicts household energy usage** based on historical trends.  
✔ **Interactive Dashboard** with visual insights.  
✔ **Data Upload & Analysis** for exploring energy consumption patterns.  
✔ **Real-time Visualizations** (Line charts, bar charts, box plots, and more).  
✔ **Comparison of predicted vs. actual values** (if historical data is available).  

---

## 🛠 **Technologies Used**
- **Python** - Core programming language.
- **Streamlit** - Web framework for interactive dashboards.
- **Pandas & NumPy** - Data manipulation and preprocessing.
- **Scikit-learn** - Machine learning model training.
- **Matplotlib & Seaborn** - Data visualization.
- **Plotly** - Interactive charts.
- **Joblib** - Model serialization (`.pkl` files).

---

## 🚀 **Project Workflow**
### **1️⃣ Data Understanding & Exploration**
- Loaded and analyzed the dataset.
- Performed **Exploratory Data Analysis (EDA)** to identify trends, correlations, and missing values.

### **2️⃣ Data Preprocessing**
- Handled missing values and data inconsistencies.
- Parsed timestamps into meaningful features (**hour, weekday, month**).
- Created rolling averages (`Rolling_Mean_3hr`, `Rolling_Mean_6hr`).
- Scaled numerical data for improved model accuracy.

### **3️⃣ Feature Engineering**
- Selected key variables impacting energy usage.
- Incorporated time-based energy trends for better forecasting.

### **4️⃣ Model Selection & Training**
- Trained multiple regression models:
  - **Linear Regression**
  - **Random Forest**
  - **Gradient Boosting**
  - **Neural Networks**
- Tuned hyperparameters for the best performance.
- Saved the final model using `joblib`.

### **5️⃣ Model Evaluation**
- Assessed model accuracy using:
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **R-Squared (R²)**
- Compared models to select the best-performing one.

### **6️⃣ Results & Insights**
- Developed an **interactive Streamlit dashboard**.
- Added **real-time visualizations** (energy usage trends, correlations).
- Allowed users to **upload their own data** and get instant predictions.
- Compared **predicted vs. actual values** when historical data is available.

---

## 📽️ **Demo Video & 👤 Author Details**
**🎥 Demo Video:** https://www.linkedin.com/posts/vasuki27_machinelearning-datascience-ai-activity-7300528832837545987-6jC1?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFWofHABP5vZ1q4SVksdeQ_qxpl9ilnOKXM  
**👨‍💻 Author:** VASUKI ARUL  
**🔗 LinkedIn:** [Vasuki Arul](https://www.linkedin.com/in/vasuki27/)  
**📅 Batch Code:** DS-C-WD-E-B29  

---

