import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime


plt.style.use('bmh') # grafiklerin stilini belirliyoruz
# Get the stock quote
df = pd.read_csv('/Users/tunahangoker/Desktop/robot/robot4/datalar/garan.csv')

#print(training_data_len)

# 0 ile 1 arasında değerler alacak şekilde ölçeklendirme yapıyoruz
scaler = MinMaxScaler(feature_range=(0, 1)) 

# ölçeklendirme işlemi
scaled_data = scaler.fit_transform(df['open'].values.reshape(-1,1)) 


timee = pd.to_datetime(df['time'])



#dataları çağırdık openları aldık
time = df['time'].values.reshape(-1,1)
time = np.array(time)
open = df['open'].values.reshape(-1,1)
open = np.array(open)
close = df['close'].values.reshape(-1,1)
close = np.array(close)
high = df['high'].values.reshape(-1,1)
high = np.array(high)
low = df['low'].values.reshape(-1,1)
low = np.array(low)
VWMA = df['VWMA'].values.reshape(-1,1)
VWMA = np.array(VWMA)
Developing_Poc = df['Developing Poc'].values.reshape(-1,1)
Developing_Poc = np.array(Developing_Poc)
Developing_VA_High = df['Developing VA High'].values.reshape(-1,1)
Developing_VA_High = np.array(Developing_VA_High)
Developing_VA_Low = df['Developing VA Low'].values.reshape(-1,1)
Developing_VA_Low = np.array(Developing_VA_Low)
Plot = df['Plot'].values.reshape(-1,1)
Plot = np.array(Plot)
Aroon_Up = df['Aroon Up'].values.reshape(-1,1)
Aroon_Up = np.array(Aroon_Up)
Aroon_Down = df['Aroon Down'].values.reshape(-1,1)
Aroon_Down = np.array(Aroon_Down)
RSI = df['RSI'].values.reshape(-1,1)
RSI = np.array(RSI)
RSI_based_MA = df['RSI-based MA'].values.reshape(-1,1)
RSI_based_MA = np.array(RSI_based_MA)
Upper_Bollinger_Band = df['Upper Bollinger Band'].values.reshape(-1,1)
Upper_Bollinger_Band = np.array(Upper_Bollinger_Band)
Lower_Bollinger_Band = df['Lower Bollinger Band'].values.reshape(-1,1)
Lower_Bollinger_Band = np.array(Lower_Bollinger_Band)
K = df['K'].values.reshape(-1,1)
K = np.array(K)
D = df['D'].values.reshape(-1,1)
D = np.array(D)

#print(RSI)

output_size = 2         # sonraki 4 degeri tahmin etmeye calis
epochs = 700            # islemi tekrar sayisi
features = 3


#time,open,high,low,close,VWMA,Developing Poc,Developing VA High,Developing VA Low,Plot,Aroon Up,Aroon Down,RSI,RSI-based MA,Upper Bollinger Band,Lower Bollinger Band,K,D

def robot1():
    a = 0
    b = 0
    c = 0
    d = [0] * 1000
    birincifiyat = [0] * 1000
    ikincifiyat = [0] * 1000
    
    for i in range(0, len(close)):

    # lstm datayi (rows, timesteps, features) formatinda ister
    # rows : ornek sayimiz
    # timesteps : zaman adimlari
    # features : ogrenme verimizin sutun sayisi
    # training verisi (rows, 1, features)
    # output verisi (rows, output_size)
    # y değerini ne kadar önceki değerlerle tahmin edeceğimizi belirler


        train_size = int(close[i])                # Verinin %70 ini egitim icin kalaninin da test icin ayiracagiz, 180

        train = close[0:train_size] #
        test = close[train_size:len(close)]
        # print("Shape : ", train.shape)
        ############## TRAIN DATA ####################
        train_x = []
        train_y = []
        for i in range(0, train_size - features - output_size):
            tmp_x = close[i:(i+features)] # burada 3 tane önceki değeri alıyoruz
            tmp_y = close[(i+features):(i+features+output_size)]  # burada 2 tane sonraki değeri alıyoruz
            train_x.append(np.reshape(tmp_x, (1, features))) # burada append ederken tmp_x'i 1,3'lük bir matrise çeviriyoruz
            train_y.append(tmp_y)

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        ########### TEST DATA ########################
        test_x = []
        test_y = []
        last = len(close) - output_size - features
        for i in range(train_size, last):
            tmp_x = close[i:(i+features)]
            tmp_y = close[(i + features):(i + features + output_size)]
            test_x.append(np.reshape(tmp_x, (1, features)))
            test_y.append(tmp_y)

        test_x = np.array(test_x)
        test_y = np.array(test_y)

        ######## Tahmin edilecek data #######################
        data_x = []
        tmp_x = close[-features:len(close)]
        data_x.append(np.reshape(tmp_x, (1, features)))
        data_x = np.array(data_x)

        return train_x, train_y, test_x, test_y, data_x
    raw_data = pd.read_csv('borsa_data/GARAN.csv')
    t = np.array(raw_data.t.values)
    y = np.array(raw_data.y.values)

    min = y.min()
    max = y.max()

    y = np.interp(y, (min, max), (-1, +1))

    x_train, y_train, x_test, y_test, data_x = robot1(y)

    model = Sequential()
    model.add(LSTM(5, input_shape=(1, features), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(7))
    model.add(Dense(output_size))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, verbose=0)

    score = model.evaluate(x_test, y_test)
    print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    model.summary()

    data_y = model.predict(data_x)

    result = np.interp(data_y, (-1, +1), (min, max)) # normalize edilen değerleri geri dönüştürüyoruz çünkü tahmin edilen değerler normalize edilmişti

    print("Gelecekteki Degerler (output_size) :", result)



#robot1()
raw_data = pd.read_csv('borsa_data/GARAN.csv')
y = np.array(raw_data.close.values)

min = y.min()
max = y.max()

y = np.interp(y, (min, max), (-1, +1))

x_train, y_train, x_test, y_test, data_x = robot1()

model = Sequential()
model.add(LSTM(5, input_shape=(1, features), return_sequences=True))
model.add(Dropout(0.25))
model.add(LSTM(7))
model.add(Dense(output_size))

model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
model.fit(x_train, y_train)

score = model.evaluate(x_test, y_test)
print("%2s: %.2f%%" % (model.metrics_names[1], score[1]*100))
model.summary()

data_y = model.predict(data_x)

result = np.interp(data_y, (-1, +1), (min, max)) # normalize edilen değerleri geri dönüştürüyoruz çünkü tahmin edilen değerler normalize edilmişti

print("Gelecekteki Degerler (output_size) :", result)




