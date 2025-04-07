import os
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
from scipy.signal import butter, filtfilt, hilbert


# Función para filtrar la señal en la banda de ripples (150-250 Hz)
def bandpass_filter_gpu(data, lowcut=150, highcut=250, fs=30000, order=4):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Convertir a arrays de CuPy para usar en la GPU
    data_cp = cp.array(data)
    b_cp = cp.array(b)
    a_cp = cp.array(a)

    # Aplicar el filtro usando CuPy y convertir de vuelta a numpy para compatibilidad con filtfilt
    return filtfilt(b_cp.get(), a_cp.get(), data_cp.get())  # Convertir a NumPy explícitamente


# Configurar directorio y archivo
file_dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection"
data_dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection/data"  # Ajusta el directorio si es necesario
filename = "datafile001.ns6"
file_path = os.path.join(data_dir, filename)

# Verificar que el archivo existe
if not os.path.exists(file_path):
    raise FileNotFoundError(f"El archivo {filename} no se encuentra en {data_dir}")

# Cargar datos con Neo
reader = neo.io.BlackrockIO(file_path)
block = reader.read_block()
signal = block.segments[0].analogsignals[0]

# Obtener datos de la señal y transponer (Canales x Muestras)
signal_data = cp.array(signal).T  # Convertir a CuPy para aprovechar la GPU
fs = int(signal.sampling_rate)

# Lista de canales a procesar (elige manualmente los índices, por ejemplo: [0, 2, 4, 6])
selected_channels = [4, 5, 6, 7]  # Modifica esta lista con los canales deseados

# Seleccionar los canales específicos
signal_data = signal_data[selected_channels]
ch_names = [f"ch_{idx + 1}" for idx in selected_channels]

# Filtrar la señal en la banda de ripples
filtered_data = cp.array([bandpass_filter_gpu(ch, fs=fs) for ch in signal_data])

# Calcular la envolvente usando Hilbert (en la GPU)
envelope = cp.abs(cp.array(hilbert(filtered_data.get())))

# Detección de ripples para cada canal
ripple_events = []
for ch_idx, ch_name in enumerate(ch_names):
    # Calcular umbral individual por canal
    threshold = cp.mean(envelope[ch_idx]) + 5 * cp.std(envelope[ch_idx])

    # Detectar puntos donde la señal supera el umbral
    ripple_mask = envelope[ch_idx] > threshold
    ripple_times = np.where(ripple_mask.get())[0] / fs  # Convertir a segundos

    # Guardar eventos solo si hay detecciones
    if len(ripple_times) > 0:
        ripple_events.append(pd.DataFrame({
            'file': filename,
            'channel': ch_name,
            'time': ripple_times
        }))

# Crear DataFrame de eventos y guardarlo
event_df = pd.concat(ripple_events, ignore_index=True) if ripple_events else pd.DataFrame(
    columns=['file', 'channel', 'time'])

csv_filename = os.path.join(file_dir, f"ripple_events_{filename}.csv")
event_df.to_csv(csv_filename, index=False)
print(f"Eventos detectados guardados en 'ripple_events_{filename}.csv'.")

# Crear DataFrame de señales para graficar
signal_df = pd.DataFrame(cp.asnumpy(signal_data.T),
                         columns=ch_names)  # Convertir a numpy para compatibilidad con pandas

# Crear la columna 'time' con el número correcto de muestras
signal_df['time'] = np.arange(len(signal_df)) / fs  # Convertir de muestras a segundos

# Configuración de la figura con 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()  # Convertir la matriz en lista para iterar más fácil

colors = ['black', 'red', 'green', 'blue']
ylim_range = (-1000, 1000)  # Mantener escala uniforme

for idx, (channel, ax) in enumerate(zip(ch_names, axes)):
    color = colors[idx % len(colors)]

    # Graficar la señal en su respectivo subplot
    ax.plot(signal_df['time'], signal_df[channel], color=color, alpha=0.7)

    # Marcar eventos de Ripple en el gráfico correspondiente
    ripple_ch_events = event_df[event_df['channel'] == channel]
    for _, event in ripple_ch_events.iterrows():
        ax.axvspan(xmin=event['time'], xmax=event['time'] + 0.05, color='cyan', alpha=0.3)

    # Ajustar límites y etiquetas
    ax.set_xlim([0, signal_df['time'].max()])
    ax.set_ylim(ylim_range)
    ax.set_title(f"Canal: {channel}")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Amplitud")
    ax.grid(True, linestyle="--", alpha=0.5)

# Ajustar diseño
plt.tight_layout()
plt.show()
