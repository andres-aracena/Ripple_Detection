import os
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
from scipy.signal import cheby2, filtfilt

# Set directory and file
file_dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection"
data_dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection/data"
filename = "datafile001.ns6"
file_path = os.path.join(data_dir, filename)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"El archivo {filename} no se encuentra en {data_dir}")

# Load data with Neo
reader = neo.io.BlackrockIO(file_path)
block = reader.read_block()
signal = block.segments[0].analogsignals[0]
signal_data = cp.array(signal).T
fs = int(signal.sampling_rate)

# Definir ventana de tiempo en segundos
start_time = 0  # segundo
end_time = 10  # segundo
start_sample = start_time * fs
end_sample = end_time * fs

# Extraer datos en la ventana de tiempo especificada
signal_data = signal_data[:, start_sample:end_sample]

# Definir los 32 canales
selected_channels = [f'ch_{i + 1}' for i in range(32)]

# Crear DataFrame para almacenar las señales filtradas
time_array = np.linspace(start_time, end_time, signal_data.shape[1])
signal_df = pd.DataFrame({'time': time_array})

# Filtrar y almacenar datos en el DataFrame
b, a = cheby2(4, 40, [70 / (0.5 * fs), 260 / (0.5 * fs)], btype='band')
filtered_signals = []

for i in range(len(selected_channels)):
    data_np = cp.asnumpy(signal_data[i])  # Convertir de cupy a numpy
    filtered_data = filtfilt(b, a, data_np)
    filtered_signals.append(filtered_data)

# Convertir lista de señales filtradas en array
filtered_signals = np.array(filtered_signals)

# Calcular promedio y desviación estándar
mean_signal = np.mean(filtered_signals, axis=0)
std_signal = np.std(filtered_signals, axis=0)

# Calcular desviación de cada canal respecto al promedio
channel_deviation = np.mean(np.abs(filtered_signals - mean_signal), axis=1)

# Seleccionar el 75% de los canales más cercanos al promedio
num_selected = int(len(selected_channels) * 0.75)
closest_channels_indices = np.argsort(channel_deviation)[:num_selected]
selected_filtered_signals = filtered_signals[closest_channels_indices]
selected_channels_names = [selected_channels[i] for i in closest_channels_indices]
print('El canal mas parecido al promedio es: ',selected_channels_names[0])

# Crear DataFrame para almacenar las señales seleccionadas
time_array = np.linspace(start_time, end_time, signal_data.shape[1])
signal_df = pd.DataFrame({'time': time_array})
signal_df['mean'] = mean_signal
for i, channel in enumerate(selected_channels_names):
    signal_df[channel] = selected_filtered_signals[i]

# Guardar solo los primeros 10 segundos en el archivo CSV
csv_duration = 5  # segundos
csv_samples = int(csv_duration * fs)

# Crear un nuevo DataFrame solo con los primeros 10 segundos
csv_df = signal_df.iloc[:csv_samples]

# Guardar los datos en un archivo CSV
csv_filename = os.path.join(file_dir, f"5s_{filename}.csv")
csv_df.to_csv(csv_filename, index=False)
print(f"Archivo guardado con los primeros 5 segundos en: {csv_filename}")


# Graficar señales seleccionadas
plt.figure(figsize=(12, 6))

for channel in selected_channels_names:
    plt.plot(signal_df['time'], signal_df[channel], alpha=0.3, color="gray")

# Graficar la señal promedio y su banda de desviación estándar
plt.plot(signal_df['time'], signal_df['mean'], label='Promedio', color="red")
plt.fill_between(signal_df['time'], signal_df['mean'] - std_signal,
                 signal_df['mean'] + std_signal, alpha=0.2, color="green", label="Desviación estándar")

# Etiquetas y leyenda
plt.title('Señales Filtradas y Promedio con Desviación Estándar')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Amplitud Normalizada')
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()