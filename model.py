import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Lambda

def build_model(window_size=10):
    model = Sequential([
        Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[window_size]),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(1),
        Lambda(lambda x: x * 100.0)
    ])
    model.compile(
        loss='huber_loss',
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=['mae']
    )
    return model

def preprocess_data(fruit_type, window_size=10):
    df = pd.read_csv('dataset/kalimati_tarkari_dataset.csv')
    df = df[df['Commodity'] == fruit_type]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Average']].resample('D').ffill()
    input_series = df['Average'].values[-window_size:]
    input_series = np.array(input_series).reshape(1, -1)
    
    return input_series
