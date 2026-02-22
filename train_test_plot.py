import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas
import yfinance as yf



#input data preparation, generating synthetic input data

n = 1000*28*28
synthetic = np.zeros(n)
temp=3000
for i in range(n):
	temp = temp + np.random.normal(0, 2.5)
	synthetic[i] = temp
	


#normalizing
noisy_values = (synthetic - np.mean(synthetic)) / np.std(synthetic)
noisy_values = noisy_values - min(noisy_values)

#shaping input into image-like format 28*28*1
sample_len=28**2

noisy_sample=np.zeros(sample_len)
x_val_noisy_r = []
x_val_pure_r = []

n_noisysamples = n // sample_len
n_oversamples = n % sample_len
n_moving = 50

for i in range(n_moving):
	moving_values = pandas.DataFrame(noisy_values).rolling(i+2, min_periods=1).mean().values
	for j in range(n_noisysamples):
		noisy_sample = noisy_values[n_oversamples + j*sample_len : n_oversamples + (j+1)*sample_len]
		pure_sample = moving_values[n_oversamples + j*sample_len: n_oversamples + (j+1)*sample_len]
		tmp_n=np.copy(noisy_sample.reshape(28,28))
		tmp_p=np.copy(pure_sample.reshape(28,28))
		x_val_noisy_r.append(tmp_n)
		x_val_pure_r.append(tmp_p)
	moving_values = pandas.DataFrame(noisy_values).ewm(i+2, min_periods=1).mean().values
	for j in range(n_noisysamples):
		noisy_sample = noisy_values[n_oversamples + j*sample_len : n_oversamples + (j+1)*sample_len]
		pure_sample = moving_values[n_oversamples + j*sample_len: n_oversamples + (j+1)*sample_len]
		tmp_n=np.copy(noisy_sample.reshape(28,28))
		tmp_p=np.copy(pure_sample.reshape(28,28))
		x_val_noisy_r.append(tmp_n)
		x_val_pure_r.append(tmp_p)



x_val_noisy_r = np.array(x_val_noisy_r)
x_val_pure_r = np.array(x_val_pure_r)

noisy_input = x_val_noisy_r.reshape((x_val_noisy_r.shape[0], x_val_noisy_r.shape[1], x_val_noisy_r.shape[2], 1))
pure_input = x_val_pure_r.reshape((x_val_pure_r.shape[0], x_val_pure_r.shape[1], x_val_pure_r.shape[2], 1))


import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.constraints import max_norm
from keras.optimizers import RMSprop


# training the auto encoder on the synthetic data
# model configuration
width, height = 28, 28
input_shape = (width, height, 1)
batch_size = 150
no_epochs = 10
max_norm_value = 2.0
validation_split = 0.2

 

model=Sequential()

#encoder
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape,padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',padding='same'))

#decoder
model.add(Conv2D(128, (3, 3), activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(1, kernel_size=(3, 3), activation='relu', padding='same'))

model.summary()

model.compile(optimizer=RMSprop(), loss='mean_squared_error')
model.fit(noisy_input, pure_input,
                epochs=no_epochs,
                batch_size=batch_size,
                validation_split=validation_split)




# testing by denoising the 'real' data

# preparing the testing data
def get_ohlc_data(ticker, start_date, interval):
    
    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

ticker = 'BTC-USD'

df = get_ohlc_data(ticker, '2025-01-01', '1h')

mean_value = np.mean(df['Close'].values)
sd_value = np.std(df['Close'].values)

noisy_values = (df['Close'].values - mean_value) / sd_value
min_noisy_value = min(noisy_values)
noisy_values = noisy_values - min_noisy_value

noisy_sample=np.zeros(sample_len)
y_val_noisy_r = []


n_noisysamples = df.shape[0] // sample_len + 1
n_remaining = sample_len - df.shape[0] % sample_len



for i in range(n_noisysamples):
	if (i+1)*sample_len <= df.shape[0]:
		noisy_sample = noisy_values[i*sample_len: (i+1)*sample_len]
	else:
		noisy_sample[:len(noisy_values[i*sample_len:])] = noisy_values[i*sample_len:,0]
	tmp_n=np.copy(noisy_sample.reshape(28,28))
	y_val_noisy_r.append(tmp_n)
	noisy_sample=np.zeros(sample_len)
	
y_val_noisy_r = np.array(y_val_noisy_r)
noisy_input = y_val_noisy_r.reshape((y_val_noisy_r.shape[0], y_val_noisy_r.shape[1], y_val_noisy_r.shape[2], 1))



#denoising the 'real' data and plotting the result


filtered = model.predict(noisy_input)

filtered = filtered + min_noisy_value
filtered = filtered * sd_value + mean_value

df['filtered'] = filtered.reshape(filtered.size,1)[:-n_remaining]
plt.figure(figsize=(10,5))
plt.plot(df['Close'],'k',alpha=0.1,label='noisy input')
plt.plot(df['filtered'],'b',alpha=1,label='filtered output')
plt.legend()
plt.xlabel('time')
plt.ylabel('closing price')
plt.show()

