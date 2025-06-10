# an-lisis_sismicos.py
```python
import numpy as np
import pandas as pd
from obspy import read
from scipy import signal
from sklearn.preprocessing import MinMaxScaler

def load_seismic_data(filepath):
    """
    Carga datos sísmicos desde un archivo SAC, MSEED u otros formatos soportados por ObsPy
    """
    st = read(filepath)
    return st

def preprocess_seismic_data(stream, lowcut=0.5, highcut=10.0, fs=100.0, order=4):
    """
    Preprocesamiento básico: filtrado y normalización
    """
    # Filtro pasa banda
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    
    processed_data = []
    for trace in stream:
        # Aplicar filtro
        filtered = signal.filtfilt(b, a, trace.data)
        
        # Normalización
        scaler = MinMaxScaler(feature_range=(-1, 1))
        normalized = scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
        
        processed_data.append(normalized)
    
    return np.array(processed_data)

def create_dataset(data, window_size=100, step_size=10):
    """
    Crea ventanas deslizantes para el entrenamiento de modelos
    """
    X = []
    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i+window_size])
    
    return np.array(X)

def split_data(X, y, test_size=0.2, val_size=0.1):
    """
    Divide los datos en conjuntos de entrenamiento, validación y prueba
    """
    from sklearn.model_selection import train_test_split
    
    # Primera división: train + temp / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    # Segunda división: train / val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size_adjusted, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```
