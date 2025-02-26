import streamlit as st
import joblib
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import plotly.express as px


# 👉 Sidebar Navigation in Box Model
# 👉 Sidebar Navigation with Project Approach Added
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Insights", "Data View", "Project Approach", "Project Info"],  # Added "Project Approach"
        icons=["bar-chart", "graph-up-arrow", "table", "list-task", "info-circle"],
        menu_icon="menu-button-wide",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#F0FFFF"},
            "icon": {"color": "#89CFF0", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#CCCCFF",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#3F00FF", "color": "white"},
        },
    )


# ✅ Load Model & Scaler Using Direct File Paths
model_path = "/Users/arul/Documents/VASUKI/projects/powerpulse_rf_model.pkl"
scaler_path = "/Users/arul/Documents/VASUKI/projects/scaler.pkl"

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    #st.sidebar.success("✅ Model & Scaler Loaded!")
else:
    st.sidebar.error("⚠️ Model or Scaler File Missing!")

# 👉 Section: Dashboard
if selected == "Dashboard":
    st.title("📊 Household Energy Usage Forecast")
    st.markdown("### 📌 Project Overview")
    st.write("- Predict household energy usage using machine learning.")
    st.write("- Gain insights into patterns and trends in energy consumption.")
    st.write("- Help optimize energy usage for cost savings.")

    st.markdown("### ⚙️ Features Used")
    st.write("- **Machine Learning Model:** Random Forest Regression.")
    st.write("- **Data Processing:** Feature scaling, missing value handling.")
    st.write("- **Visualization:** Charts and graphs to analyze trends.")

    st.markdown("### 🎯 Project Objectives")
    st.write("- Identify high energy usage periods.")
    st.write("- Help users optimize their electricity consumption.")
    st.write("- Provide real-time insights for smart energy management.")

    st.markdown("### ✅ Advantages")
    st.write("- Reduces electricity costs by analyzing usage patterns.")
    st.write("- Supports sustainability by promoting efficient energy consumption.")
    st.write("- Helps in predicting and avoiding power outages.")


# 👉 Section: Insights
elif selected == "Insights":
    st.title("📈 Energy Insights")
    st.markdown("### 🔍 Exploring key trends and predictions.")

    # 🔹 File Upload for Dataset
    st.markdown("### 📂 Upload Household Energy Usage Data")
    uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.write("📌 Data Preview:")
        st.write(df.head())  # Show first 5 rows to confirm data
    
        # 🔹 Line Chart - Energy Consumption Over Time
        st.markdown("### 📈 Energy Usage Over Time")
        fig1 = px.line(df, x="Hour", y="Energy_Usage", color="Weekday",
                       title="Hourly Energy Consumption by Weekday")
        st.plotly_chart(fig1, use_container_width=True)

        # 🔹 Bar Chart - Monthly Energy Usage Comparison
        st.markdown("### 📊 Monthly Energy Usage Comparison")
        monthly_avg = df.groupby("Month")["Energy_Usage"].mean().reset_index()
        fig2 = px.bar(monthly_avg, x="Month", y="Energy_Usage",
                      title="Average Energy Usage per Month", color="Energy_Usage")
        st.plotly_chart(fig2, use_container_width=True)

        # 🔹 Bar Chart - Energy Usage by Weekday
        st.markdown("### 📅 Energy Usage by Weekday")
        weekday_avg = df.groupby("Weekday")["Energy_Usage"].mean().reset_index()
        weekday_avg["Weekday"] = weekday_avg["Weekday"].replace(
            {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
             4: "Friday", 5: "Saturday", 6: "Sunday"}
        )
        fig3 = px.bar(weekday_avg, x="Weekday", y="Energy_Usage",
                      title="Average Energy Usage per Weekday", color="Energy_Usage")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("⚠️ Please upload a CSV file to view insights.")



# 👉 Section: Data View
elif selected == "Data View":
    st.title("📂 Data Overview")
    st.markdown("### 📊 Explore Household Energy Usage Data")

    # 🔹 File Upload for Custom Dataset
    uploaded_file = st.file_uploader("📤 Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")

        # 🔹 Display Dataset Info
        st.markdown("### 📌 Dataset Information")
        st.write(f"- **Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write(df.dtypes)

        # 🔹 Show Missing Values
        st.markdown("### ⚠️ Missing Values")
        missing_values = df.isnull().sum()
        st.write(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

        # 🔹 Display Summary Statistics
        st.markdown("### 📊 Summary Statistics")
        st.write(df.describe())

        # 🔹 Column Descriptions (Example)
        st.markdown("### 📌 Column Descriptions")
        column_info = {
            "Hour": "The hour of the day (0-23).",
            "Weekday": "Day of the week (0=Monday, 6=Sunday).",
            "Month": "Month of the year (1-12).",
            "Rolling_Mean_3hr": "Average energy usage over the last 3 hours.",
            "Rolling_Mean_6hr": "Average energy usage over the last 6 hours.",
            "Energy_Usage": "Total energy consumption in kWh."
        }
        for col, desc in column_info.items():
            if col in df.columns:
                st.write(f"**{col}:** {desc}")

        # 🔹 Data Visualization

        # Line Chart: Energy Usage Over Time
        st.markdown("### 📈 Energy Usage Trends")
        fig1 = px.line(df, x="Hour", y="Energy_Usage", color="Weekday",
                       title="Hourly Energy Consumption by Weekday")
        st.plotly_chart(fig1, use_container_width=True)

        # Box Plot: Outlier Detection
        st.markdown("### 📊 Detecting Outliers in Energy Usage")
        fig2 = px.box(df, y="Energy_Usage", title="Energy Usage Outliers")
        st.plotly_chart(fig2, use_container_width=True)

        # Pie Chart: Distribution of Energy Usage by Weekday
        st.markdown("### 📅 Energy Usage by Weekday")
        weekday_counts = df["Weekday"].value_counts().reset_index()
        weekday_counts.columns = ["Weekday", "Count"]
        weekday_counts["Weekday"] = weekday_counts["Weekday"].replace(
            {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
             4: "Friday", 5: "Saturday", 6: "Sunday"}
        )
        fig3 = px.pie(weekday_counts, names="Weekday", values="Count",
                      title="Energy Usage Distribution by Weekday")
        st.plotly_chart(fig3, use_container_width=True)

        # 🔹 Download Processed Data
        st.markdown("### 📥 Download Processed Data")
        st.download_button("Download CSV", df.to_csv(index=False), "processed_data.csv", "text/csv")

    else:
        st.warning("⚠️ Please upload a CSV file to explore data.")


# 👉 Section: Project Info
elif selected == "Project Info":
    st.title("ℹ️ About the Project")
    st.markdown("### 📢 Why We Built This Project?")
    st.write("- To analyze household energy consumption patterns.")
    st.write("- To provide insights for reducing electricity costs.")
    st.write("- To contribute to efficient energy management.")

    st.markdown("### 📚 Technologies Used")
    st.write("- **Programming:** Python, Streamlit")
    st.write("- **Machine Learning:** Scikit-learn (Random Forest)")
    st.write("- **Data Processing:** Pandas, NumPy")
    st.write("- **Visualization:** Matplotlib, Seaborn")

    st.markdown("### 💡 Future Enhancements")
    st.write("- Add real-time energy monitoring.")
    st.write("- Integrate with IoT devices for live tracking.")
    st.write("- Implement advanced forecasting models.")

# 👉 Section: Project Approach
elif selected == "Project Approach":
    st.title("📌 Project Approach")
    st.markdown("### 🔍 Step-by-Step Process for Household Energy Forecasting")

    # 🔹 Data Understanding & Exploration
    st.subheader("📊 Data Understanding & Exploration")
    st.write("- Load and analyze the dataset to understand its structure.")
    st.write("- Perform **Exploratory Data Analysis (EDA)** to detect patterns, correlations, and anomalies.")

    # 🔹 Data Preprocessing
    st.subheader("⚙️ Data Preprocessing")
    st.write("- Handle missing values and data inconsistencies.")
    st.write("- Parse timestamps into meaningful features (e.g., **day, time, season**).")
    st.write("- Create additional features like **daily averages, peak usage hours, rolling averages**.")
    st.write("- Normalize or scale the data for better model performance.")

    # 🔹 Feature Engineering
    st.subheader("🛠️ Feature Engineering")
    st.write("- Identify key variables that impact energy usage.")
    st.write("- Incorporate external factors (e.g., **weather conditions**) if available.")

    # 🔹 Model Selection & Training
    st.subheader("📈 Model Selection & Training")
    st.write("- Split the dataset into **training and testing sets**.")
    st.write("- Train different regression models:")
    st.markdown("""
        - 🔹 **Linear Regression**
        - 🔹 **Random Forest**
        - 🔹 **Gradient Boosting**
        - 🔹 **Neural Networks**
    """)
    st.write("- Perform **hyperparameter tuning** to enhance model accuracy.")

    # 🔹 Model Evaluation
    st.subheader("📉 Model Evaluation")
    st.write("- Assess model performance using:")
    st.markdown("""
        - ✅ **Root Mean Squared Error (RMSE)**
        - ✅ **Mean Absolute Error (MAE)**
        - ✅ **R-Squared (R²)**
    """)
    st.write("- Compare different models and select the best-performing one.")

    # 🔹 Results & Insights
    st.subheader("📊 Results & Insights")
    st.write("- Develop visualizations to illustrate **trends, consumption patterns, feature importance**.")
    st.write("- Generate actionable insights to **optimize energy consumption**.")

    #st.success("✅ Project Approach Successfully Implemented!")

