# Bitcoin Price Prediction using LSTM

This project uses a Deep Learning LSTM model to predict Bitcoin closing prices using historical data. The model is built with TensorFlow and Keras.

---

## 📂 Dataset

- Source: [Kaggle]([https://finance.yahoo.com/](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data))
- File: `BTC-USD.csv`
- Features Used: `Close`, `Open`, `High`, `Volume`

---

## 🛠️ Technologies Used

- Python
- Pandas & NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## 📈 Model Overview

The LSTM model is trained using the past 60 days of closing prices to predict the next day's price.

### 🧠 LSTM Model Code (Short Snippet):

```python
model = Sequential()
model.add(LSTM(100, return_sequences=False, input_shape=(60, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=75, batch_size=32)
