from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Load and prepare time series data
time_series_data = pd.read_csv("data.csv", encoding='ISO-8859-1')
if 'Revenue' not in time_series_data.columns:
    time_series_data['Revenue'] = time_series_data['Quantity'] * time_series_data['UnitPrice']
time_series_data['InvoiceDate'] = pd.to_datetime(time_series_data['InvoiceDate'])
time_series_data = time_series_data.groupby(time_series_data['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
time_series_data.rename(columns={'InvoiceDate': 'Date', 'Revenue': 'DailyRevenue'}, inplace=True)
time_series_data['Date'] = pd.to_datetime(time_series_data['Date'])
time_series_data.set_index('Date', inplace=True)
time_series_data = time_series_data.asfreq('D')
time_series_data['DailyRevenue'].fillna(0, inplace=True)

# Train ARIMA model
model = ARIMA(time_series_data['DailyRevenue'], order=(1, 1, 1))
model_fit = model.fit()
prediction_cache = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/forecast", methods=["POST"])
def forecast():
    days_to_predict = int(request.form.get("days", 7))

    if days_to_predict in prediction_cache:
        predictions = prediction_cache[days_to_predict]
    else:
        forecast_result = model_fit.get_forecast(steps=days_to_predict)
        forecast_index = pd.date_range(
            start=time_series_data.index[-1] + pd.Timedelta(days=1),
            periods=days_to_predict,
            freq='D'
        )
        predictions = pd.DataFrame({
            'Date': forecast_index,
            'ForecastedRevenue': forecast_result.predicted_mean
        })
        prediction_cache[days_to_predict] = predictions

    return jsonify(predictions.to_dict(orient="records"))


@app.route("/performance")
def performance():
    """
    Serve a performance metrics visualization as an image.
    """
    # Create a simple performance plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_data.index, time_series_data['DailyRevenue'], label='Actual', color='blue')
    plt.plot(time_series_data.index, model_fit.fittedvalues, label='Fitted', linestyle='--', color='orange')
    plt.legend()
    plt.title("Model Performance: Actual vs Fitted")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.tight_layout()

    # Save the plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    # Return the plot as a response
    return send_file(buf, mimetype="image/png", as_attachment=False, download_name="performance.png")


@app.route('/favicon.ico')
def favicon():
    return '', 204


if __name__ == "__main__":
    app.run(debug=True)
