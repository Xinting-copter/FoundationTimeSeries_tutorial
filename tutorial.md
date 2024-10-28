## LLM-related tutorials

Creating a tutorial for using the Foundation time series model involves explaining how to install, configure, and apply the model for forecasting tasks. Here's a structured guide to get started:

# Tutorial: How to Use the Foundation Time Series Model

This tutorial will guide you through installing, configuring, and applying the Foundation time series model for forecasting time series data.

![SADA system](path_to_image/.png)
*Figure 1: SADA measurement Over Time*

### Prerequisites
Before you begin, ensure you have the following tools installed:
- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

You can install these dependencies by running:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 1. Installing Foundation Time Series Model

First, you need to install the Foundation model package. If it’s not available on PyPI, download or clone the repository from the model's official GitHub repository:

```bash
git clone https://github.com/your-repository/foundation-ts-model.git
cd foundation-ts-model
pip install .
```

If it’s available through pip, install it directly:
```bash
pip install foundation-ts-model
```

### 2. Preparing Your Dataset

Foundation works with time series data, so you must prepare your dataset in a format suitable for time series analysis. Typically, this means a DataFrame where:
- The index is a `DatetimeIndex`
- Columns represent different variables or features of the time series.

For example, if you are working with stock prices:
```python
import pandas as pd

# Load your data (make sure the Date column is parsed as a datetime object)
df = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')

# Display the first few rows
print(df.head())
```

Your data should look like this:
```
              Open    High     Low   Close   Volume
Date
2024-01-01   100.1   105.3    98.4   104.0   1000000
2024-01-02   104.5   107.2   102.1   106.1   1200000
...
```

### 3. Configuring the Foundation Model

Foundation time series model requires specific configuration, such as the forecast horizon and feature selection. You can specify these through its parameters.

```python
from foundation_ts_model import FoundationModel

# Define model parameters
forecast_horizon = 10  # Predict 10 time steps into the future
lookback_window = 30   # Use the last 30 time steps for prediction

# Instantiate the Foundation model
model = FoundationModel(forecast_horizon=forecast_horizon, lookback_window=lookback_window)
```

You can customize other parameters depending on your needs, such as seasonal factors, trend inclusion, or multivariate forecasting.

### 4. Splitting the Dataset

For training and evaluation, split the dataset into training and test sets.

```python
# Split the data into train and test sets
train_size = int(len(df) * 0.8)  # 80% training, 20% test
train, test = df[:train_size], df[train_size:]
```

### 5. Test the Model

Test the Foundation model on the test data.

```python
# Train the model
model.fit(train)
```

You can also print the training performance or visualize the loss history if the model supports it:
```python
model.plot_training_loss()  # If supported
```

### 6. Fine-tuning 

Once the model is trained, you can make predictions for the forecast horizon.

```python
# Predict future values
predictions = model.predict(test.index)

# Compare predictions with the actual values
comparison = pd.DataFrame({'Actual': test['Close'], 'Predicted': predictions})
print(comparison.head())
```

### 7. Evaluating the Model

Evaluate the model's performance using standard time series metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE):

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(test['Close'], predictions)
rmse = mean_squared_error(test['Close'], predictions, squared=False)

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
```

### 8. Visualizing the Results

Finally, plot the actual vs. predicted values to visualize the model's performance.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(test.index, test['Close'], label='Actual')
plt.plot(test.index, predictions, label='Predicted', linestyle='--')
plt.title('Actual vs. Predicted Time Series')
plt.xlabel('Reference Time Index')
plt.ylabel('Shell temperature')
plt.legend()
plt.show()
```

### 9. Advanced Features

Foundation offers several advanced features:
- **Multivariate Forecasting**: Include multiple variables (e.g., Open, High, Low) for better prediction.
- **Seasonality Adjustments**: Handle seasonality by adding explicit seasonal components.
- **Auto-tuning**: Automatically find the best model parameters.

Refer to the official documentation for more advanced configurations and options.

### Conclusion

This tutorial has introduced you to the basics of using the Foundation time series model. With your model trained and evaluated, you can apply it to various time series forecasting tasks, from stock prices to demand forecasting.

For further reading, consult the [official documentation](https://link-to-foundation-docs.com).
