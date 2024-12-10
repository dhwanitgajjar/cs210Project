# Sales Forecasting and Visualization for E-commerce using ARIMA

## Setup Instructions

### 1. Creating the Virtual Environment

To isolate project dependencies, create a virtual environment:

```
python -m venv venv
```

### 2. Activating the Virtual Environment

Activate the virtual environment with the following command:

```
source ./venv/bin/activate
```

### 3. Installing Required Libraries

Install all necessary libraries by running:

```
pip install -r requirements.txt
```

### 4. Running the Prediction Notebook

Open and execute the `sales_forecasting_CS210.ipynb` Jupyter Notebook cell by cell to train and evaluate the model.

**Project Overview**

This project aims to develop a real-time forecasting and visualization system for an e-commerce platform. It uses the ARIMA model for sales predictions and provides an interactive Flask-based dashboard to monitor performance metrics and gain actionable insights.

**Project Files**
* app.py: Flask application for real-time forecasting and performance visualization.
* data.csv: Cleaned e-commerce transactional data for time series analysis.
* index.html: Front-end for the Flask dashboard, styled for user-friendly interaction.
* Project.ipynb: Jupyter Notebook for exploratory data analysis and model validation.
* Final report.docx: Detailed documentation of the project's scope, implementation, and results.

**Dashboard Features**
1) Forecasting: Enter the number of days to predict future sales and view the results dynamically.
2) Performance Metrics: Visualize actual vs. fitted revenue trends using an interactive chart.

**How to access the dashboard**
![image](https://github.com/user-attachments/assets/7e74c5ab-c51f-4139-998f-61a5c704e59b)
Run the app.py file. Install the necessary libraries. When the terminal opens up, look for : **Running on http://127.0.0.1:5000**

Hold Ctrl and left Click on it to access the dashboard.



