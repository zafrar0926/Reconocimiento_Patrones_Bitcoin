import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = ['datos_procesados/features', 'modelos/scalers']
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

# Crear directorios
crear_directorios()

# === 1. CARGAR DATOS BASE ===
print("Cargando datos...")
df = pd.read_csv('dataset_btc.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# === 2. CREAR TARGETS FUTUROS (para evitar data leakage) ===
df['target_t1'] = np.log(df['close'].shift(-1) / df['close'])
df['target_t6'] = np.log(df['close'].shift(-6) / df['close'])
df['target_t12'] = np.log(df['close'].shift(-12) / df['close'])

# Eliminar columnas de retorno ya precalculado que podrían tener leakage
df.drop(columns=[
    'log_return_1h', 'log_return_6h',
    'log_return_12h', 'log_return_24h',
    'log_return_168h'
], inplace=True, errors='ignore')

# === 3. DIVISIÓN TEMPORAL ===
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print(f"Tamaño conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño conjunto de prueba: {len(df_test)}")

# === 4. FUNCIONES DE FEATURE ENGINEERING ===
def crear_caracteristicas_tecnicas(df):
    df_features = df.copy()
    df_features['SMA_7'] = SMAIndicator(close=df['close'], window=7, fillna=True).sma_indicator()
    df_features['SMA_21'] = SMAIndicator(close=df['close'], window=21, fillna=True).sma_indicator()
    df_features['EMA_7'] = EMAIndicator(close=df['close'], window=7, fillna=True).ema_indicator()
    df_features['EMA_21'] = EMAIndicator(close=df['close'], window=21, fillna=True).ema_indicator()

    macd = MACD(close=df['close'], fillna=True)
    df_features['MACD'] = macd.macd()
    df_features['MACD_signal'] = macd.macd_signal()
    df_features['MACD_diff'] = macd.macd_diff()

    df_features['RSI'] = RSIIndicator(close=df['close'], fillna=True).rsi()

    if 'high' in df.columns and 'low' in df.columns:
        stoch = StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'], fillna=True
        )
    else:
        df_features['rolling_high'] = df['close'].rolling(window=24, center=False).max()
        df_features['rolling_low'] = df['close'].rolling(window=24, center=False).min()
        stoch = StochasticOscillator(
            high=df_features['rolling_high'],
            low=df_features['rolling_low'],
            close=df['close'],
            fillna=True
        )

    df_features['Stoch_k'] = stoch.stoch()
    df_features['Stoch_d'] = stoch.stoch_signal()

    bollinger = BollingerBands(close=df['close'], fillna=True)
    df_features['BB_high'] = bollinger.bollinger_hband()
    df_features['BB_low'] = bollinger.bollinger_lband()
    df_features['BB_mid'] = bollinger.bollinger_mavg()

    df_features.drop(columns=['rolling_high', 'rolling_low'], errors='ignore', inplace=True)
    return df_features

def crear_caracteristicas_retornos(df):
    df_features = df.copy()
    # Usar rolling de close para estimar volatilidad
    for ventana in [12, 24, 48, 168]:
        df_features[f'volatility_{ventana}h'] = df['close'].pct_change().rolling(
            window=ventana,
            center=False,
            min_periods=1
        ).std()
    
    for periodo in [12, 24, 168]:
        df_features[f'momentum_{periodo}h'] = df['close'].pct_change(periods=periodo)

    return df_features

def crear_caracteristicas_temporales(df):
    df_features = df.copy()
    df_features['hour'] = df.index.hour
    df_features['day'] = df.index.day
    df_features['weekday'] = df.index.weekday
    df_features['month'] = df.index.month

    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['weekday_sin'] = np.sin(2 * np.pi * df_features['weekday'] / 7)
    df_features['weekday_cos'] = np.cos(2 * np.pi * df_features['weekday'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)

    return df_features

def normalizar_features(df_train, df_test=None):
    scaler = StandardScaler()
    columnas_numericas = df_train.select_dtypes(include=['float64', 'int64']).columns
    columnas_numericas = [col for col in columnas_numericas if not col.startswith('target_')]

    df_train_norm = df_train.copy()
    df_train_norm[columnas_numericas] = scaler.fit_transform(df_train[columnas_numericas])

    # Guardar el scaler
    joblib.dump(scaler, 'modelos/scalers/standard_scaler.joblib')

    if df_test is not None:
        df_test_norm = df_test.copy()
        df_test_norm[columnas_numericas] = scaler.transform(df_test[columnas_numericas])
        return df_train_norm, df_test_norm

    return df_train_norm

# === 5. APLICAR FEATURE ENGINEERING ===
print("\nCreando características...")

df_train = crear_caracteristicas_tecnicas(df_train)
df_train = crear_caracteristicas_retornos(df_train)
df_train = crear_caracteristicas_temporales(df_train)

df_test = crear_caracteristicas_tecnicas(df_test)
df_test = crear_caracteristicas_retornos(df_test)
df_test = crear_caracteristicas_temporales(df_test)

# === 6. LIMPIEZA FINAL ===
df_train = df_train.dropna()
df_test = df_test.dropna()

# === 7. NORMALIZAR ===
print("\nNormalizando características...")
df_train_norm, df_test_norm = normalizar_features(df_train, df_test)

# === 8. GUARDAR ===
print("\nGuardando datos procesados...")
df_train_norm.to_csv('datos_procesados/features/btc_features_train.csv')
df_test_norm.to_csv('datos_procesados/features/btc_features_test.csv')

# Guardar lista de características
with open('datos_procesados/features/lista_caracteristicas.txt', 'w') as f:
    f.write("Características creadas:\n")
    for col in df_train_norm.columns:
        f.write(f"- {col}\n")
    f.write(f"\nDimensiones del dataset de entrenamiento: {df_train_norm.shape}\n")
    f.write(f"Dimensiones del dataset de prueba: {df_test_norm.shape}\n")

print("\n¡Proceso completado!")
print("Los resultados han sido guardados en:")
print("- datos_procesados/features/btc_features_train.csv")
print("- datos_procesados/features/btc_features_test.csv")
print("- datos_procesados/features/lista_caracteristicas.txt")
print("- modelos/scalers/standard_scaler.joblib")