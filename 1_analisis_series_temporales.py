import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def crear_directorios():
    """Crear estructura de directorios necesaria"""
    directorios = ['datos_procesados', 'graficos/analisis_temporal']
    for dir in directorios:
        Path(dir).mkdir(parents=True, exist_ok=True)

# Crear directorios
crear_directorios()

# Cargar datos
df = pd.read_csv('dataset_btc.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Los retornos logarítmicos ya vienen calculados en el dataset
# Usaremos log_return_1h como nuestro retorno principal

def analisis_estacionariedad(serie):
    """Realiza prueba de Dickey-Fuller aumentada"""
    result = adfuller(serie.dropna())
    print('Resultados prueba ADF:')
    print(f'Estadístico ADF: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Valores críticos:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    # Guardar resultados en un archivo
    with open('datos_procesados/resultados_adf.txt', 'w') as f:
        f.write('Resultados prueba ADF:\n')
        f.write(f'Estadístico ADF: {result[0]}\n')
        f.write(f'p-value: {result[1]}\n')
        f.write('Valores críticos:\n')
        for key, value in result[4].items():
            f.write(f'\t{key}: {value}\n')

def plot_serie_temporal(serie, titulo, guardar_como=None):
    """Grafica una serie temporal con su media móvil"""
    plt.figure(figsize=(15, 6))
    plt.plot(serie)
    plt.plot(serie.rolling(window=30).mean(), 'r', label='Media móvil 30 días')
    plt.title(titulo)
    plt.legend()
    if guardar_como:
        plt.savefig(guardar_como)
        plt.close()
    else:
        plt.show()

# 1. Análisis de la serie temporal
print("Análisis Estadístico Descriptivo:")
descripcion = df['log_return_1h'].describe()
print(descripcion)

# Guardar descripción estadística
descripcion.to_csv('datos_procesados/descripcion_estadistica.csv')

# 2. Visualización de la serie temporal
plot_serie_temporal(
    df['close'],
    'Precio de Cierre del Bitcoin',
    'graficos/analisis_temporal/precio_cierre.png'
)
plot_serie_temporal(
    df['log_return_1h'],
    'Retornos Logarítmicos del Bitcoin (1h)',
    'graficos/analisis_temporal/retornos_log.png'
)

# 3. Análisis de estacionariedad
print("\nAnálisis de Estacionariedad para Retornos Logarítmicos:")
analisis_estacionariedad(df['log_return_1h'].dropna())

# 4. Descomposición de la serie
descomposicion = seasonal_decompose(df['close'], period=24)  # 24 horas para datos horarios
plt.figure(figsize=(15, 12))
plt.subplot(411)
plt.plot(descomposicion.observed)
plt.title('Observado')
plt.subplot(412)
plt.plot(descomposicion.trend)
plt.title('Tendencia')
plt.subplot(413)
plt.plot(descomposicion.seasonal)
plt.title('Estacionalidad')
plt.subplot(414)
plt.plot(descomposicion.resid)
plt.title('Residuos')
plt.tight_layout()
plt.savefig('graficos/analisis_temporal/descomposicion_temporal.png')
plt.close()

# 5. Análisis de autocorrelación
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
plot_acf(df['log_return_1h'].dropna(), ax=ax1, lags=40)
ax1.set_title('Función de Autocorrelación (ACF)')
plot_pacf(df['log_return_1h'].dropna(), ax=ax2, lags=40)
ax2.set_title('Función de Autocorrelación Parcial (PACF)')
plt.tight_layout()
plt.savefig('graficos/analisis_temporal/autocorrelacion.png')
plt.close()

# 6. Análisis de volatilidad
df['volatilidad'] = df['log_return_1h'].rolling(window=24).std()  # Ventana de 24 horas
plt.figure(figsize=(15, 6))
plt.plot(df['volatilidad'])
plt.title('Volatilidad Móvil (24 horas)')
plt.savefig('graficos/analisis_temporal/volatilidad.png')
plt.close()

# Guardar resultados procesados
df.to_csv('datos_procesados/btc_processed.csv')

print("\n¡Proceso completado!")
print("Los resultados han sido guardados en:")
print("- datos_procesados/btc_processed.csv")
print("- datos_procesados/descripcion_estadistica.csv")
print("- datos_procesados/resultados_adf.txt")
print("- graficos/analisis_temporal/*.png") 