import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Model file path
MODEL_PATH = "drowsiness_model.h5"

def train_lstm_model():
    """Train an LSTM model for drowsiness detection"""
    X_train = np.array([[0.22, 0.4], [0.20, 0.5], [0.18, 0.6], [0.15, 0.7]])
    y_train = np.array([0, 0, 1, 1])
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(1, 2)),
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)

    model.save(MODEL_PATH)
    print("✅ Model saved!")
    return model

# Load or train the model
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # Recompile
    print("✅ Model loaded & compiled!")
else:
    print("🚀 Training new model...")
    model = train_lstm_model()

def get_trained_model():
    """Returns the trained LSTM model"""
    return model
