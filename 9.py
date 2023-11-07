import os
from tensorflow import keras
from tensorflow.keras import layers
from numpy import random
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
from sklearn.preprocessing import MaxAbsScaler


######################################################################################################################################################################################################
plt.style.use('bmh') # grafiklerin stilini belirliyoruz
# Get the stock quote
data = pd.read_csv('/Users/tunahangoker/Desktop/robot/robot4/datalar/garan.csv', header=None, names=["time","open","high","low","close","Basis","Upper","Lower","VWMA","Developing Poc"
                                                                   ,"Developing VA High","Developing VA Low","Plot","RSI Cloud Lead","RSI Cloud Base",
                                                                   "RSI Overbought ","Signal RSI Overbought","RSI Oversold","Signal RSI Oversold","Baseline",
                                                                   "Cloud Flip Bullish","Cloud Flip Bearish","RENKLİRSI","RSI","RSI-based MA","Upper Bollinger Band"
                                                                   ,"Lower Bollinger Band","K","D"])


######################################################################################################################################################################################################




time = data['time'].values.reshape(-1,1) 
time = np.array(time)
open = data['open'].values.reshape(-1,1) # 1D array
open = np.array(open)
open = np.transpose(open)
high = data['high'].values.reshape(-1,1)
high = np.array(high)
low = data['low'].values.reshape(-1,1)
low = np.array(low)
close = data['close'].values.reshape(-1,1)
close = np.array(close)
Basis = data['Basis'].values.reshape(-1,1)
Basis = np.array(Basis)
Upper = data['Upper'].values.reshape(-1,1)
Upper = np.array(Upper)
Lower = data['Lower'].values.reshape(-1,1)
Lower = np.array(Lower)
VWMA = data['VWMA'].values.reshape(-1,1)
VWMA = np.array(VWMA)
Developing_Poc = data['Developing Poc'].values.reshape(-1,1)
Developing_Poc = np.array(Developing_Poc)
Developing_VA_High = data['Developing VA High'].values.reshape(-1,1)
Developing_VA_High = np.array(Developing_VA_High)
Developing_VA_Low = data['Developing VA Low'].values.reshape(-1,1)
Developing_VA_Low = np.array(Developing_VA_Low)
Plot = data['Plot'].values.reshape(-1,1)
Plot = np.array(Plot)
RSI_Cloud_Lead = data['RSI Cloud Lead'].values.reshape(-1,1)
RSI_Cloud_Lead = np.array(RSI_Cloud_Lead)
RSI_Cloud_Base = data['RSI Cloud Base'].values.reshape(-1,1)
RSI_Cloud_Base = np.array(RSI_Cloud_Base)
RSI_Overbought = data['RSI Overbought '].values.reshape(-1,1)
RSI_Overbought = np.array(RSI_Overbought)
Signal_RSI_Overbought = data['Signal RSI Overbought'].values.reshape(-1,1)
Signal_RSI_Overbought = np.array(Signal_RSI_Overbought)
RSI_Oversold = data['RSI Oversold'].values.reshape(-1,1)
RSI_Oversold = np.array(RSI_Oversold)
Signal_RSI_Oversold = data['Signal RSI Oversold'].values.reshape(-1,1)
Signal_RSI_Oversold = np.array(Signal_RSI_Oversold)
Baseline = data['Baseline'].values.reshape(-1,1)
Baseline = np.array(Baseline)
Cloud_Flip_Bullish = data['Cloud Flip Bullish'].values.reshape(-1,1)
Cloud_Flip_Bullish = np.array(Cloud_Flip_Bullish)
Cloud_Flip_Bearish = data['Cloud Flip Bearish'].values.reshape(-1,1)
Cloud_Flip_Bearish = np.array(Cloud_Flip_Bearish)
RENKLİRSI = data['RENKLİRSI'].values.reshape(-1,1)
RENKLİRSI = np.array(RENKLİRSI)
RSI = data['RSI'].values.reshape(-1,1)
RSI = np.array(RSI)
RSI = np.transpose(RSI)
RSI_based_MA = data['RSI-based MA'].values.reshape(-1,1)
RSI_based_MA = np.array(RSI_based_MA)
Upper_Bollinger_Band = data['Upper Bollinger Band'].values.reshape(-1,1)
Upper_Bollinger_Band = np.array(Upper_Bollinger_Band)
Lower_Bollinger_Band = data['Lower Bollinger Band'].values.reshape(-1,1)
Lower_Bollinger_Band = np.array(Lower_Bollinger_Band)
K = data['K'].values.reshape(-1,1)
K = np.array(K)
D = data['D'].values.reshape(-1,1)
D = np.array(D)
######################################################################################################################################################################################################


open_dataa = open.reshape(-1, 1)  # 1D array

class CustomActivation(layers.Layer):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs)  # Ortalamayı hesapla
        #std = tf.math.reduce_std(inputs)  # Standart sapmayı hesapla
        output = (inputs - mean) 
        output = tf.abs(output[:])
        return output




import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class StockPredictionModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(StockPredictionModel, self).__init__()
        self.dense1 = layers.Dense(hidden_units, activation=CustomActivation())
        self.dense2 = layers.Dense(hidden_units, activation=CustomActivation())
        self.dense3 = layers.Dense(hidden_units, activation=CustomActivation())
        self.dense4 = layers.Dense(hidden_units, activation=CustomActivation())
        self.output_layer = layers.Dense(32)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        output = self.output_layer(x)
        return output

    def predict_next_price(self, input_data):
        predicted_prices = []
        for i in range(len(input_data)):
            if i >= 14:
                previous_data = input_data[i-14:i]
                
                # Aykırı değerleri kaldırma işlemi
                predicted_price = self.call(np.array([previous_data]))

                
                predicted_prices.append(predicted_price[0][0])
            else:
                predicted_prices.append(None)
        return predicted_prices

'''
# Kullanım örneği
model = StockPredictionModel(hidden_units=32)

predictions = model.predict_next_price(open)
print(predictions)

'''















# Kullanım örneği:
# Kullanım örneği:
hidden_units = 32
stock_prediction_model = StockPredictionModel(32)


# Örnek veri oluşturma
open_data = data['open'].values.reshape(-1, 1)  # 1D array
open_data = np.array(open_data)







# Batch oluşturma diziler burdan ayarlanıyo 
batch_size = 32
time_steps = 1
dataset = tf.data.Dataset.from_tensor_slices(open_data)
dataset = dataset.window(time_steps+1, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(time_steps+1))
dataset = dataset.map(lambda window: (window[:-1], window[-1]))
dataset = dataset.batch(batch_size, drop_remainder=True)

RSI_data = data['RSI'].values.reshape(-1,1)
RSI_data = np.array(RSI_data)


RSIbatch_size = 32
RSItime_steps = 1
RSIdataset = tf.data.Dataset.from_tensor_slices(RSI_data)
RSIdataset = RSIdataset.window(time_steps+1, shift=1, drop_remainder=True)
RSIdataset = RSIdataset.flat_map(lambda window: window.batch(time_steps+1))
RSIdataset = RSIdataset.map(lambda window: (window[:-1], window[-1]))
RSIdataset = RSIdataset.batch(batch_size, drop_remainder=True)


















'''
# Model derleme
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
stock_prediction_model.compile(optimizer=optimizer, loss=loss_fn)

# Model eğitimi
epochs = 0
stock_prediction_model.fit(dataset, epochs=epochs)
'''
# Tahmin yapma
'''for batch_input, batch_target in dataset:
    predictions = stock_prediction_model(batch_input)
    print(predictions)'''
    


concatenated_data = open_dataa
predicted_prices = stock_prediction_model.predict_next_price(concatenated_data)

print(predicted_prices)


































# Ölçüt hesaplama (örnek olarak ortalama mutlak hata - Mean Absolute Error)
mae = tf.keras.metrics.MeanAbsoluteError()
for batch_input, batch_target in dataset:
    predictions = stock_prediction_model(batch_input)
    mae.update_state(batch_target, predictions)
mean_absolute_error = mae.result()
print("Mean Absolute Error:", mean_absolute_error)
'''numpy_predictions = predictions.numpy()
print(numpy_predictions[33])
'''
#istediğin datayı çağırma metodu
def get_data_point(dataset, index):
    index= index - 1
    batch_index = index // batch_size
    data_index = index % batch_size
    
    batch_input, batch_target = next(iter(dataset.skip(batch_index).take(1)))
    desired_data = batch_input[data_index]
    
    return desired_data



def RSI_get_data_point(RSIdataset, index):
    index= index - 1
    batch_index = index // batch_size
    data_index = index % batch_size
    
    batch_input, batch_target = next(iter(RSIdataset.skip(batch_index).take(1)))
    desired_data = batch_input[data_index]
    
    return desired_data






#get_data_point(dataset, 2200) kullanımı 


#1420466400,9.58,9.64,9.57,9.59,9.458500000000008,9.585665246822469,9.331334753177547,9.48948684794804,NaN,NaN,NaN,1,79.3174688163295,63.346977145598096,70,75,30,25,50,NaN,NaN,71.13089854041255,71.13089854041255,62.01485350492438,72.42572398817927,51.603983021669485,100.00000000000004,100.00000000000004

kontrol = True


stock_prediction_model.summary()
################################################################################################################################################################

if kontrol !=True:



 


    b = 0
    c = 0
    birincifiyat = [0] * 1000
    ikincifiyat = [0] * 1000
    for i in range(0, len(close)):
            if RSI[i] <= 41:
            
                if RSI[i] + 1.63 <= Lower_Bollinger_Band[i]:
                    
                    print("                         ilk sinyal")
                    
                    print("RSI : ", RSI[i])
                    print("low bolinger : ", Lower_Bollinger_Band[i])
                    
                    print("Alım yapılacak zaman: ", time[i])
                    print("Alım yapılacak fiyat: ", close[i])
                    print(" i = ",i)
                    print("K : ", K[i])
                    print("D : ", D[i])
                    while True:
                        if K[i] + 2.01 >= D[i] :
                            
                            print(" -----------------------------------------------------------------------------------------------")
                            print("               ikinci sinyal")
                            print("RSI : ", RSI[i])
                            print("low bolinger : ", Lower_Bollinger_Band[i])
                            print("K : ", K[i])
                            print("D : ", D[i])
                            print(" i = ",i)
                            print("Alım yapılacak zaman: ", time[i])
                            print("Alım yapılacak fiyat: ", close[i])
                            print(" -----------------------------------------------------------------------------------------------")
                            if i >= 14:
                                
                                fiyatlistesi1 = []
                                for j in range(i-14, i+14):
                                    fiyatlistesi1.append(close[j])


                                
                                plt.style.use('bmh')  # Set the style of the plots

                            # Create a figure with 2 rows and 1 column, and set the size
                                fig, ax = plt.subplots(2, 1, figsize=(8, 16))

                            # Plot the first close prices
                                
                                ax[0].set_title('Close Price History')
                                ax[0].set_xlabel('Time', fontsize=18)
                                ax[0].set_ylabel('Close Price USD ($)', fontsize=18)
                                ax[0].plot(fiyatlistesi1, color = 'black', label = 'birincifiyat')
                                ax[0].legend(loc='upper left')
                            # Plot the second close price
                                ax[1].set_title('Close Price History')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
                                ax[1].set_xlabel('Time', fontsize=18)
                                ax[1].set_ylabel('Close Price USD ($)', fontsize=18)
                                ax[1].plot(close, color = 'black', label = 'close')
                                ax[1].legend(loc='upper left')
                                #plt.show()
                        

                                
            
                            break
                        else:
                            i=i+1
                            
                            
                else:
                    i = i + 1











