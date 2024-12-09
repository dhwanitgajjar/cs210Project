from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load prepared time series data
time_series_data = pd.read_csv("data.csv", encoding='ISO-8859-1')

# Ensure the Revenue column is calculated
if 'Revenue' not in time_series_data.columns:
    time_series_data['Revenue'] = time_series_data['Quantity'] * time_series_data['UnitPrice']

# Convert InvoiceDate to datetime and prepare the time series
time_series_data['InvoiceDate'] = pd.to_datetime(time_series_data['InvoiceDate'])
time_series_data = time_series_data.groupby(time_series_data['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
time_series_data.rename(columns={'InvoiceDate': 'Date', 'Revenue': 'DailyRevenue'}, inplace=True)
time_series_data['Date'] = pd.to_datetime(time_series_data['Date'])
time_series_data.set_index('Date', inplace=True)

# Global ARIMA model (pre-trained or fit on the dataset)
model = ARIMA(time_series_data['DailyRevenue'], order=(1, 1, 1))
model_fit = model.fit()

# Caching predictions to optimize performance
prediction_cache = {}


@app.route("/")
def index():
    """
    Render the dashboard for real-time visualization and reporting.
    """
    return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Real-time forecasting API endpoint.
    """
    # Parse the request data
    days_to_predict = int(request.form.get("days", 7))

    # Use cache if available
    if days_to_predict in prediction_cache:
        predictions = prediction_cache[days_to_predict]
    else:
        # Generate forecast
        forecast_result = model_fit.forecast(steps=days_to_predict)
        dates = pd.date_range(start=time_series_data.index[-1], periods=days_to_predict + 1, freq="D")[1:]
        predictions = pd.DataFrame({"Date": dates, "ForecastedRevenue": forecast_result})
        prediction_cache[days_to_predict] = predictions

    # Convert to JSON for API response
    return jsonify(predictions.to_dict(orient="records"))


@app.route("/performance")
def performance():
    """
    Serve a performance metrics visualization.
    """
    # Create a simple performance plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_data.index, time_series_data['DailyRevenue'], label='Actual')
    plt.plot(time_series_data.index, model_fit.fittedvalues, label='Fitted', linestyle='--')
    plt.legend()
    plt.title("Model Performance: Actual vs Fitted")
    plt.xlabel("Date")
    plt.ylabel("Revenue")

    # Save the plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
    buf.close()

    return render_template("index.html", plot_url=plot_url)


if __name__ == "__main__":
    app.run(debug=True)
