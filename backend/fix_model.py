from pathlib import Path

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

print("🛠️ Generating local-compatible universal_stock_model.keras file...")

lookback = 60
features_count = 4

model = Sequential([
    Input(shape=(lookback, features_count)),
    Bidirectional(LSTM(32, return_sequences=True)),
    Dropout(0.1),
    LSTM(16),
    Dense(1)
])
model.compile(optimizer='adam', loss='huber')

model_path = Path(__file__).resolve().parent / "universal_stock_model.keras"
model.save(model_path)
print(f"✅ SUCCESS! Saved to {model_path}")
