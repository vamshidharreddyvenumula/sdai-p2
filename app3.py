from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load LSTM model
lstm_stock_model = load_model('lstm_model.h5')  # Update this path
scaler = MinMaxScaler(feature_range=(0, 1))  # Assuming MinMaxScaler was used during model training

# Load ARIMA models
with open('model_apl_arima.pkl', 'rb') as pkl:
    model_apl_arima = pickle.load(pkl)

with open('model_fb_arima.pkl', 'rb') as pkl:
    model_fb_arima = pickle.load(pkl)
def create_prediction_dates(start_date, periods):
    return [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(periods)]

def plot_to_base64(plt_figure):
    """Converts a matplotlib figure to a base64 encoded string."""
    img = BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Check if file is uploaded
            file = request.files['file']
            if file:
                # Read the uploaded file
                df = pd.read_csv(file)
                print("DataFrame:", df.head())
                last_date = pd.to_datetime(df['date'].iloc[-1])
                future_dates = [last_date + timedelta(days=i) for i in range(1, 11)]

                exog_data_apl = df.iloc[:10, [0]]
                exog_data_fb = df.iloc[:10, [0]]
                print("Exogenous data for APL:", exog_data_apl)
                print("Exogenous data for FB:", exog_data_fb)
                
                # Assuming you want to use the first column for predictions
                # Modify this part according to your CSV structure
                predictions_apl = model_apl_arima.predict(n_periods=10, exogenous=exog_data_apl)
                predictions_fb = model_fb_arima.predict(n_periods=10, exogenous=exog_data_fb)

                
                df = df[['close']]  # Assuming 'close' column is used for prediction
                scaled_data = scaler.fit_transform(df.values)

                # Reshape data for LSTM model
                X_test = []
                for i in range(40, len(scaled_data)):
                    X_test.append(scaled_data[i-40:i, 0])

                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                # Make prediction
                prediction = lstm_stock_model.predict(X_test)

                # Convert prediction to a suitable format
                first_10_predictions = prediction[:10].flatten().tolist()
                # Plotting the predictions
                fig_lstm, ax_lstm = plt.subplots()
                ax_lstm.plot(future_dates, first_10_predictions)
                ax_lstm.set_title('LSTM Predictions')
                ax_lstm.set_xlabel('Date')
                ax_lstm.set_ylabel('Prediction')
                plot_url_lstm = plot_to_base64(fig_lstm)

                fig_apl, ax_apl = plt.subplots()
                ax_apl.plot(future_dates, predictions_apl)
                ax_apl.set_title('Apple Stock Predictions')
                ax_apl.set_xlabel('Date')
                ax_apl.set_ylabel('Prediction')
                plot_url_apl = plot_to_base64(fig_apl)

                fig_fb, ax_fb = plt.subplots()
                ax_fb.plot(future_dates, predictions_fb)
                ax_fb.set_title('Facebook Stock Predictions')
                ax_fb.set_xlabel('Date')
                ax_fb.set_ylabel('Prediction')
                plot_url_fb = plot_to_base64(fig_fb)
                return render_template('result3.html', dates=future_dates, predictions_lstm=first_10_predictions,  plot_url_lstm=plot_url_lstm, 
                               plot_url_apl=plot_url_apl, plot_url_fb=plot_url_fb, predictions_apl=predictions_apl, predictions_fb=predictions_fb)

    except Exception as e:
        print("Error:", e)  # Print the exception for debugging
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
