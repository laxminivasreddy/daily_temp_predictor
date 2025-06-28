#Import the Libraries

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

#Load the dataset into the Dataframe
#parse_date is used to convert the date column into datetime objects

df = pd.read_csv("daily_minimum_temps.csv", parse_dates = ["Date"], index_col = "Date")

df.head()

df.shape

# Converting the string values to Numeric values by removing the double quotes and if string value does not makes any sense or it
#We convert that value to null.

df["Temp"]= pd.to_numeric(df["Temp"], errors = "coerce")

df = df.dropna()

#Normalize the Features.

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df["Temp"].values.reshape(-1,1))

#sequence length
seq_length = 30

def create_sequences(data_scaled, seq_length):
    X = []
    y = []
    for i in range(len(data_scaled) - seq_length):
        X.append(data_scaled[i:i+seq_length])
        y.append(data_scaled[i+seq_length])
    return np.array(X), np.array(y)

#calling the function and storing the values in list x and y

X,y = create_sequences(data_scaled, seq_length)

#divide the dataset into train and split
X_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, shuffle = False)

#building the RNN model
model = Sequential([
    LSTM(64,activation = "relu",input_shape = (seq_length,1)),
    Dense(1) #since output will be a single 
])

#compile the model
model.compile(optimizer = "adam",loss = "mse")

model.fit(X_train,y_train,epochs = 20,batch_size = 32)

#make predictions
y_pred_scaled = model.predict(x_test)

#inverse transform the scaled data
y_pred_scaled = np.clip(y_pred_scaled,0,1)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_test_actaul = scaler.inverse_transform(y_test)

#predict the next day tempurature
last_sequence = data_scaled[-seq_length:].reshape(1,seq_length,1)
next_temp_scaled = model.predict(last_sequence)
next_temp_scaled = np.clip(next_temp_scaled,0,1)
next_day_temp = scaler.inverse_transform(next_temp_scaled)

print("next day temperature is",next_day_temp)


joblib.dump(scaler, "Scaler.pkl")

model.save("Temp_Predictor_Model.h5")
print("The Model has been saved")