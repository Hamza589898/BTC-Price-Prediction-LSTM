# import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# import csv.file
df = pd.read_csv("BTC-USD.csv")

# Min-Max-Scaling
scale = MinMaxScaler()
scale_data = scale.fit_transform(df[["Close",'Open','High','Volume']])
scale_df = pd.DataFrame(scale_data,columns=['Close','Open','High','Volume'])

# scale close column
close_scaler = MinMaxScaler()
scale_close = close_scaler.fit_transform(df[['Close']])

X = []
y = []

# arranging X and y for DL
sequence_length = 60
for i in range(sequence_length,len(scale_close)):
    X.append(scale_close[i-sequence_length:i, 0])
    y.append(scale_close[i,0])

X = np.array(X)
y = np.array(y)

# Arranging data for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Training Model
X_train,X_Test,y_train,y_Test=train_test_split(X,y,test_size=0.2,random_state=42,shuffle=False)

# Making DL Model
model = Sequential()
model.add(LSTM(100, return_sequences=False, input_shape=(sequence_length, 1)))
model.add(Dense(1))  # ✅ FIXED: Output sirf Close price

# compile model
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(X_train,y_train,epochs=75,batch_size=32)

# Predict
prediction = model.predict(X_Test)

# Convert Min-Max-Scaling in normal numbers
prediction_inv = close_scaler.inverse_transform(prediction.reshape(-1,1))
y_test_inv = close_scaler.inverse_transform(y_Test.reshape(-1,1))

# Plot for visulization
plt.figure(figsize=(10,6))
plt.plot(y_test_inv , label = "Actual Price",color='blue')
plt.plot(prediction_inv,label = "Predicted Price",color = 'red')
plt.title("Actual vs Predicted BTC Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

# Print last 10 days output
print("Predicted BTC Prices (last 10 days):")
for i, price in enumerate(prediction_inv[-10:]):
    print(f"Day {i+1}: ${price[0]:.2f}")

# Print R2 score
r2 = r2_score(y_test_inv,prediction_inv)
print(f"R2 score : {r2:.4f}")
