import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import warnings
from ta.trend import CCIIndicator, ADXIndicator
from ta.momentum import WilliamsRIndicator, TSIIndicator
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

if 'high' not in df.columns or 'low' not in df.columns:
    print("Columnas 'high' y/o 'low' no disponibles. Generando columnas sintéticas.")
    df['high'] = df['close'].rolling(window=24).max()
    df['low'] = df['close'].rolling(window=24).min()


# === 2. CREAR TARGETS FUTUROS ===
print("Creando variables objetivo...")
df['target_t1'] = np.log(df['close'].shift(-1) / df['close'])
df['target_t6'] = np.log(df['close'].shift(-6) / df['close'])
df['target_t12'] = np.log(df['close'].shift(-12) / df['close'])

# === 3. CREAR RETORNOS PASADOS (features conocidos en t) ===
print("Creando retornos pasados...")
df['return_t1'] = np.log(df['close'] / df['close'].shift(1))  # Retorno más reciente (t-1 a t)
df['return_t2'] = np.log(df['close'].shift(1) / df['close'].shift(2))  # Retorno t-2 a t-1
df['return_t3'] = np.log(df['close'].shift(2) / df['close'].shift(3))  # Retorno t-3 a t-2

# === 4. DIVISIÓN TEMPORAL ===
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

print(f"Tamaño conjunto de entrenamiento: {len(df_train)}")
print(f"Tamaño conjunto de prueba: {len(df_test)}")

# === 5. FUNCIONES DE FEATURE ENGINEERING ===
def crear_caracteristicas_tecnicas(df):
    """
    Crear características técnicas usando SOLO información pasada.
    Todos los indicadores se calculan hasta t-1 y luego se desplazan
    para asegurar que solo usamos información conocida en el momento de la predicción.
    """
    df_features = df.copy()
    
    # Precios desplazados para cálculos (solo usamos precios hasta t-1)
    close_past = df['close'].shift(1)
    
    # Medias Móviles
    df_features['SMA_7'] = SMAIndicator(close=close_past, window=7, fillna=True).sma_indicator()
    df_features['SMA_21'] = SMAIndicator(close=close_past, window=21, fillna=True).sma_indicator()
    df_features['EMA_7'] = EMAIndicator(close=close_past, window=7, fillna=True).ema_indicator()
    df_features['EMA_21'] = EMAIndicator(close=close_past, window=21, fillna=True).ema_indicator()

    # MACD
    macd = MACD(close=close_past, fillna=True)
    df_features['MACD'] = macd.macd()
    df_features['MACD_signal'] = macd.macd_signal()
    df_features['MACD_diff'] = macd.macd_diff()

    # RSI
    df_features['RSI'] = RSIIndicator(close=close_past, fillna=True).rsi()

    # Stochastic
    df_features['rolling_high'] = df['close'].shift(1).rolling(window=24, center=False).max()
    df_features['rolling_low'] = df['close'].shift(1).rolling(window=24, center=False).min()
    stoch = StochasticOscillator(
        high=df_features['rolling_high'],
        low=df_features['rolling_low'],
        close=close_past,
        fillna=True
    )
    df_features['Stoch_k'] = stoch.stoch()
    df_features['Stoch_d'] = stoch.stoch_signal()

    # Bollinger Bands
    bollinger = BollingerBands(close=close_past, fillna=True)
    df_features['BB_high'] = bollinger.bollinger_hband()
    df_features['BB_low'] = bollinger.bollinger_lband()
    df_features['BB_mid'] = bollinger.bollinger_mavg()

    # Media móvil de 24h (tendencia corta)
    df_features['rolling_mean_24h'] = df['close'].shift(1).rolling(window=24).mean()

    # Media móvil de 168h (tendencia semanal)
    df_features['rolling_mean_168h'] = df['close'].shift(1).rolling(window=168).mean()

    # Diferencia entre precio y media móvil
    df_features['trend_diff_24h'] = df['close'].shift(1) - df_features['rolling_mean_24h']
    df_features['trend_diff_168h'] = df['close'].shift(1) - df_features['rolling_mean_168h']


        # === Indicadores adicionales ===
    # CCI
    df_features['CCI'] = CCIIndicator(close=close_past, high=df['high'].shift(1), low=df['low'].shift(1), window=20, fillna=True).cci()
  
    # Williams %R
    df_features['WilliamsR'] = WilliamsRIndicator(high=df['high'].shift(1), low=df['low'].shift(1), close=close_past, lbp=14, fillna=True).williams_r()

    # TSI
    df_features['TSI'] = TSIIndicator(close=close_past, fillna=True).tsi()

    df_features['trend_3'] = df['close'].shift(1) - df['close'].shift(3)
    df_features['trend_up'] = (df_features['trend_3'] > 0).astype(int)

    df_features['swing_high'] = (df_features['high'].shift(1) > df_features['high'].shift(2)) & (df_features['high'].shift(1) > df_features['high'].shift(0))
    df_features['swing_low'] = (df_features['low'].shift(1) < df_features['low'].shift(2)) & (df_features['low'].shift(1) < df_features['low'].shift(0))

    df_features['hammer'] = ((df_features['high'] - df_features['close']) > 2 * (df_features['close'] - df_features['low'])).astype(int)
    df_features['price_speed'] = df_features['close'] - df_features['close'].shift(3)

    # Limpiar columnas temporales
    df_features.drop(columns=['rolling_high', 'rolling_low'], errors='ignore', inplace=True)
    

    return df_features


def crear_caracteristicas_retornos(df):
    """
    Crear características basadas en retornos usando SOLO información pasada.
    """
    df_features = df.copy()
    
    # Retornos pasados ya calculados anteriormente
    returns_past = np.log(df['close'] / df['close'].shift(1))
    
    # Volatilidad usando solo retornos pasados
    for ventana in [12, 24, 48, 168]:
        df_features[f'volatility_{ventana}h'] = returns_past.shift(1).rolling(
            window=ventana,
            center=False,
            min_periods=1
        ).std()
    
    # Momentum usando precios pasados
    for periodo in [12, 24, 168]:
        df_features[f'momentum_{periodo}h'] = np.log(
            df['close'].shift(1) / df['close'].shift(periodo + 1)
        )

    return df_features

def crear_caracteristicas_temporales(df):
    """Crear características temporales (no necesitan ajuste por data leakage)"""
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

# === 6. APLICAR FEATURE ENGINEERING ===
print("\nCreando características...")

df_train = crear_caracteristicas_tecnicas(df_train)
df_train = crear_caracteristicas_retornos(df_train)
df_train = crear_caracteristicas_temporales(df_train)

df_test = crear_caracteristicas_tecnicas(df_test)
df_test = crear_caracteristicas_retornos(df_test)
df_test = crear_caracteristicas_temporales(df_test)

# === 7. LIMPIEZA Y VERIFICACIÓN ===
# Eliminar columnas que no deberían ser features
columnas_a_eliminar = ['close', 'vwap']  # Precios directos
columnas_a_eliminar.extend([col for col in df_train.columns if col.startswith('log_return_')])  # Log returns originales

print("\nEliminando columnas con posible data leakage:", columnas_a_eliminar)
df_train = df_train.drop(columns=columnas_a_eliminar, errors='ignore')
df_test = df_test.drop(columns=columnas_a_eliminar, errors='ignore')

# Verificar columnas antes de continuar
print("\nColumnas finales que serán features:")
feature_cols = [col for col in df_train.columns if not col.startswith('target_')]
for col in sorted(feature_cols):
    print(f"- {col}")

# Eliminar filas con NaN
df_train = df_train.dropna()
df_test = df_test.dropna()

# === 8. NORMALIZAR ===
print("\nNormalizando características...")
df_train_norm, df_test_norm = normalizar_features(df_train, df_test)

# === 9. GUARDAR ===
print("\nGuardando datos procesados...")
df_train_norm.to_csv('datos_procesados/features/btc_features_train.csv')
df_test_norm.to_csv('datos_procesados/features/btc_features_test.csv')

# Guardar lista de características y explicación del proceso
with open('datos_procesados/features/lista_caracteristicas.txt', 'w') as f:
    f.write("=== CARACTERÍSTICAS CREADAS ===\n\n")
    f.write("1. RETORNOS PASADOS:\n")
    f.write("- return_t1: Retorno del período t-1 a t\n")
    f.write("- return_t2: Retorno del período t-2 a t-1\n")
    f.write("- return_t3: Retorno del período t-3 a t-2\n\n")
    
    f.write("2. INDICADORES TÉCNICOS (todos calculados con precios hasta t-1):\n")
    f.write("- SMA_7, SMA_21: Medias móviles simples\n")
    f.write("- EMA_7, EMA_21: Medias móviles exponenciales\n")
    f.write("- MACD, MACD_signal, MACD_diff: Indicadores MACD\n")
    f.write("- RSI: Índice de fuerza relativa\n")
    f.write("- Stoch_k, Stoch_d: Oscilador estocástico\n")
    f.write("- BB_high, BB_low, BB_mid: Bandas de Bollinger\n\n")
    f.write("- CCI: Commodity Channel Index (desviación del precio respecto a su media)\n")
   
    f.write("- WilliamsR: Oscilador de sobrecompra/sobreventa\n")
    f.write("- TSI: True Strength Index (momentum)\n")

    f.write("5. CARACTERÍSTICAS DE TENDENCIA (calculadas con precios hasta t-1):\n")
    f.write("- rolling_mean_24h, rolling_mean_168h: Media móvil de 24 y 168 horas\n")
    f.write("- trend_diff_24h, trend_diff_168h: Diferencia entre el precio y su media móvil\n\n")

    
    f.write("3. CARACTERÍSTICAS DE VOLATILIDAD Y MOMENTUM:\n")
    f.write("- volatility_Xh: Desviación estándar de retornos pasados\n")
    f.write("- momentum_Xh: Retornos acumulados en ventanas pasadas\n\n")
    
    f.write("4. CARACTERÍSTICAS TEMPORALES:\n")
    f.write("- Variables cíclicas de hora, día, semana y mes\n\n")
    
    f.write("=== DIMENSIONES ===\n")
    f.write(f"Dataset de entrenamiento: {df_train_norm.shape}\n")
    f.write(f"Dataset de prueba: {df_test_norm.shape}\n\n")
    
    f.write("=== LISTA COMPLETA DE FEATURES ===\n")
    for col in sorted(df_train_norm.columns):
        if not col.startswith('target_'):
            f.write(f"- {col}\n")

print("\n¡Proceso completado!")
print("Los resultados han sido guardados en:")
print("- datos_procesados/features/btc_features_train.csv")
print("- datos_procesados/features/btc_features_test.csv")
print("- datos_procesados/features/lista_caracteristicas.txt")
print("- modelos/scalers/standard_scaler.joblib")