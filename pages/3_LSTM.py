import streamlit as st
import pandas as pd
import numpy as np
import plost
import pickle
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Load the training data
with open('training_data.pkl', 'rb') as training_data_file:
    x_train, y_train, scaler = pickle.load(training_data_file)

# Streamlit app
st.title('Stock Price Prediction App')

# Sidebar for user input
st.sidebar.header('User Input')

# Collect user input for prediction
n_years = st.sidebar.slider('Number of Years for Prediction', 1, 5, 1)

# Perform prediction
# Replace this with your actual prediction logic
predicted_prices = np.random.rand(252 * n_years) * 1000  # Placeholder, replace with actual predictions

# Assuming you have the actual stock prices (replace actual_prices with your actual data)
actual_prices = np.random.rand(252 * n_years) * 1000  # Placeholder, replace with actual prices

# Calculate metrics
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
mse = mean_squared_error(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

# Create a DataFrame for the predicted prices
predicted_data = pd.DataFrame({
    'Date': pd.date_range(pd.Timestamp.today(), periods=len(predicted_prices)),
    'Predicted_Close_Price': predicted_prices
})

st.sidebar.subheader('Line chart parameters')
plot_data = st.sidebar.multiselect('Select data', ['Predicted_Close_Price'], ['Predicted_Close_Price'])
plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

tab1, tab2 = st.tabs(["DATASET", "VISUALISASI MAP"])

with tab1:
    # Show the predicted prices
    st.subheader('Predicted Stock Prices')
    st.write(predicted_data)

with tab2:
    # Row A
    st.markdown('### Metrics')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RMSE", round(rmse, 2))
    col2.metric("MAE", round(mae, 2))
    col3.metric("MSE", round(mse, 2))
    col4.metric("MAPE", round(mape, 2))

    with st.expander("⬇ HEATMAP:"):
        
        # Use the correct column names and data in the plost.time_hist function
        plost.time_hist(
            data=predicted_data,
            date='Date',  # Replace with the actual date column name
            x_unit='week',
            y_unit='day',
            color='Predicted_Close_Price',  # Replace with the appropriate column from your data
            aggregate='median',
            legend=None,
            height=345,
            use_container_width=True)

    # Row C
    with st.expander("### Line chart"):
        st.line_chart(predicted_data.set_index('Date')[plot_data], height=plot_height)
