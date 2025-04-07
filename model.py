import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

def parse_csv(filename):
    # Чтение данных с правильными разделителями
    df = pd.read_csv(filename, 
                    delimiter=';',
                    decimal=',',
                    names=['kzt','usd','tyr'],  # Задаем имена колонок
                    skiprows=1)  # Пропускаем заголовок
    return df

def prepare_data(data, n_steps):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i, :])  # Используем все три валюты как фичи
        y.append(scaled_data[i, 0])  # Целевая переменная - KZT/RUB (первая колонка)
        
    return np.array(X), np.array(y), scaler

def build_model(n_steps, n_features):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(n_steps, n_features)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    # Загрузка данных
    data = parse_csv('dataset/dataset.csv')
    
    # Проверка данных
    print("Первые 5 строк данных:")
    print(data.head())
    print("\nИнформация о данных:")
    print(data.info())
    
    # Параметры модели
    n_steps = 14
    test_ratio = 0.2
    
    # Подготовка данных
    X, y, scaler = prepare_data(data.values, n_steps)
    
    # Разделение данных
    split = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Создание и обучение модели
    model = build_model(n_steps, X.shape[2])
    history = model.fit(X_train, y_train,
                        epochs=100,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        verbose=1)
    
    # Предсказание
    predicted = model.predict(X_test)
    
     # Обратное преобразование масштаба
    # Для предсказанных данных
    dummy_matrix_pred = np.zeros((len(predicted), data.shape[1]))
    dummy_matrix_pred[:, 0] = predicted.flatten()
    predicted_rates = scaler.inverse_transform(dummy_matrix_pred)[:, 0]

    # Для фактических данных
    dummy_matrix_actual = np.zeros((len(y_test), data.shape[1]))
    dummy_matrix_actual[:, 0] = y_test.flatten()
    actual_rates = scaler.inverse_transform(dummy_matrix_actual)[:, 0]

    # Визуализация
    plt.figure(figsize=(14, 6))
    plt.plot(actual_rates, label='Actual KZT/RUB', color='blue', alpha=0.7)
    plt.plot(predicted_rates, label='Predicted KZT/RUB', color='orange', linestyle='--')
    plt.title('KZT/RUB Exchange Rate Prediction')
    plt.xlabel('Time Steps')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid(True)

    # Сохраняем график в файл вместо показа
    plt.savefig('kzt_prediction.png', bbox_inches='tight')  # Сохранение в текущую директорию
    plt.close()  # Закрываем фигуру для освобождения памяти

if __name__ == '__main__':
    main()