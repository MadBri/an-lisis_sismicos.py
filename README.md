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
```python
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, 
                                    Dense, Dropout, Flatten, TimeDistributed)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from utils.preprocessing import (load_seismic_data, preprocess_seismic_data, 
                                create_dataset, split_data)

# Configuración
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Parámetros
WINDOW_SIZE = 200  # Tamaño de la ventana en muestras
STEP_SIZE = 50     # Paso del ventaneo
BATCH_SIZE = 32
EPOCHS = 100

def load_and_preprocess_data(data_dir):
    """
    Carga y preprocesa todos los datos sísmicos en el directorio
    """
    all_data = []
    labels = []
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.sac') or filename.endswith('.mseed'):
            filepath = os.path.join(data_dir, filename)
            
            # Cargar datos
            stream = load_seismic_data(filepath)
            
            # Preprocesar (aquí puedes añadir más lógica según tus necesidades)
            processed = preprocess_seismic_data(stream)
            
            # Asumimos que cada archivo es una clase diferente (ajustar según tus datos)
            label = os.path.splitext(filename)[0]
            
            # Crear ventanas
            windows = create_dataset(processed[0], WINDOW_SIZE, STEP_SIZE)
            
            all_data.extend(windows)
            labels.extend([label] * len(windows))
    
    # Convertir a numpy array
    X = np.array(all_data)
    y = np.array(labels)
    
    # Codificar etiquetas
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Convertir a one-hot encoding
    y_onehot = tf.keras.utils.to_categorical(y_encoded)
    
    return X, y_onehot, le

def build_cnn_model(input_shape, num_classes):
    """
    Construye un modelo CNN para clasificación sísmica
    """
    model = Sequential([
        Conv1D(32, 5, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def build_lstm_model(input_shape, num_classes):
    """
    Construye un modelo LSTM para clasificación sísmica
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Construye un modelo híbrido CNN-LSTM
    """
    model = Sequential([
        TimeDistributed(Conv1D(32, 3, activation='relu'), input_shape=input_shape),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Conv1D(64, 3, activation='relu')),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Flatten()),
        LSTM(64),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_models(data_dir, output_dir):
    """
    Entrena los tres modelos y guarda los resultados
    """
    # Cargar y preprocesar datos
    X, y, label_encoder = load_and_preprocess_data(data_dir)
    
    # Asegurar que X tenga la forma correcta (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Obtener formas de entrada y número de clases
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(output_dir, 'best_model.h5'), 
                        monitor='val_loss', save_best_only=True)
    ]
    
    # Entrenar CNN
    print("Entrenando modelo CNN...")
    cnn_model = build_cnn_model(input_shape, num_classes)
    cnn_history = cnn_model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    cnn_model.save(os.path.join(output_dir, 'cnn_model.h5'))
    
    # Entrenar LSTM
    print("\nEntrenando modelo LSTM...")
    lstm_model = build_lstm_model(input_shape, num_classes)
    lstm_history = lstm_model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    lstm_model.save(os.path.join(output_dir, 'lstm_model.h5'))
    
    # Entrenar CNN-LSTM (necesita reshape diferente)
    print("\nEntrenando modelo CNN-LSTM...")
    # Reshape para CNN-LSTM (samples, subsequences, timesteps, features)
    X_train_cnn_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_val_cnn_lstm = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
    
    cnn_lstm_model = build_cnn_lstm_model((1, input_shape[0], input_shape[1]), num_classes)
    cnn_lstm_history = cnn_lstm_model.fit(
        X_train_cnn_lstm, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val_cnn_lstm, y_val),
        callbacks=callbacks
    )
    cnn_lstm_model.save(os.path.join(output_dir, 'cnn_lstm_model.h5'))
    
    # Evaluar modelos en el conjunto de prueba
    print("\nEvaluando modelos...")
    
    # Evaluar CNN
    cnn_loss, cnn_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"CNN - Precisión en prueba: {cnn_acc:.4f}")
    
    # Evaluar LSTM
    lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)
    print(f"LSTM - Precisión en prueba: {lstm_acc:.4f}")
    
    # Evaluar CNN-LSTM
    X_test_cnn_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    cnn_lstm_loss, cnn_lstm_acc = cnn_lstm_model.evaluate(X_test_cnn_lstm, y_test, verbose=0)
    print(f"CNN-LSTM - Precisión en prueba: {cnn_lstm_acc:.4f}")
    
    # Guardar el label encoder para uso futuro
    import pickle
    with open(os.path.join(output_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    return {
        'cnn': {'model': cnn_model, 'history': cnn_history},
        'lstm': {'model': lstm_model, 'history': lstm_history},
        'cnn_lstm': {'model': cnn_lstm_model, 'history': cnn_lstm_history}
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrena modelos CNN y LSTM para datos sísmicos.')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directorio con los datos sísmicos')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directorio para guardar los modelos entrenados')
    
    args = parser.parse_args()
    
    train_models(args.data_dir, args.output_dir)
```
```python
import os
import numpy as np
import tensorflow as tf
from utils.preprocessing import load_seismic_data, preprocess_seismic_data, create_dataset
import pickle

def load_model_and_encoder(model_dir):
    """
    Carga el modelo y el label encoder
    """
    # Cargar el mejor modelo (o puedes cargar uno específico)
    model_path = os.path.join(model_dir, 'best_model.h5')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'cnn_lstm_model.h5')  # fallback
    
    model = tf.keras.models.load_model(model_path)
    
    # Cargar el label encoder
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
    
    return model, label_encoder

def predict_seismic_event(filepath, model, label_encoder, window_size=200):
    """
    Realiza una predicción sobre un nuevo archivo sísmico
    """
    # Cargar y preprocesar datos
    stream = load_seismic_data(filepath)
    processed = preprocess_seismic_data(stream)
    
    # Crear ventanas (usamos solo el primer canal si hay múltiples)
    windows = create_dataset(processed[0], window_size, step_size=window_size)
    
    # Reshape para el modelo
    if len(model.input_shape) == 3:  # CNN o LSTM
        X_new = windows.reshape(windows.shape[0], windows.shape[1], 1)
    elif len(model.input_shape) == 4:  # CNN-LSTM
        X_new = windows.reshape(windows.shape[0], 1, windows.shape[1], 1)
    
    # Hacer predicciones
    predictions = model.predict(X_new)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    
    # Obtener probabilidades
    probabilities = np.max(predictions, axis=1)
    
    # Agregar resultados por ventana
    results = []
    for i in range(len(predicted_labels)):
        results.append({
            'window': i,
            'predicted_class': predicted_labels[i],
            'probability': float(probabilities[i]),
            'all_probabilities': predictions[i].tolist()
        })
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Realiza predicciones con modelos entrenados.')
    parser.add_argument('--filepath', type=str, required=True,
                       help='Ruta al archivo sísmico para predecir')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directorio con los modelos entrenados')
    
    args = parser.parse_args()
    
    # Cargar modelo y encoder
    model, label_encoder = load_model_and_encoder(args.model_dir)
    
    # Realizar predicción
    results = predict_seismic_event(args.filepath, model, label_encoder)
    
    # Mostrar resultados
    print("\nResultados de la predicción:")
    for result in results:
        print(f"Ventana {result['window']}:")
        print(f"  Clase predicha: {result['predicted_class']}")
        print(f"  Probabilidad: {result['probability']:.4f}")
        print("  Probabilidades por clase:")
        for i, prob in enumerate(result['all_probabilities']):
            print(f"    {label_encoder.classes_[i]}: {prob:.4f}")
        print()
```

