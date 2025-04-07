import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, cheby1, cheby2, ellip, filtfilt, freqz, impulse, dlti, dimpulse
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import neo

# Funciones de métricas
def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

def calculate_pearson_correlation(signal, original_signal):
    return pearsonr(signal, original_signal)[0]

def calculate_signal_distortion(original_signal, filtered_signal):
    return np.sum((original_signal - filtered_signal) ** 2)

def calculate_sdr(original_signal, filtered_signal):
    noise = original_signal - filtered_signal
    signal_power = np.mean(original_signal ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

# Set directory and file
file_dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection"
data_dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection/data"
filename = "datafile001.ns6"
file_path = os.path.join(data_dir, filename)

reader = neo.io.BlackrockIO(file_path)
block = reader.read_block()
signal = block.segments[0].analogsignals[0]
signal_data = np.array(signal.magnitude).T
fs = int(signal.sampling_rate.magnitude)

# Seleccionar canal
selected_channel = 24
signal_data = signal_data[selected_channel]

# Verificar si la señal tiene NaN o Inf antes de procesarla
if np.any(np.isnan(signal_data)) or np.any(np.isinf(signal_data)):
    raise ValueError("La señal original contiene valores NaN o Inf. Verifica los datos antes de filtrar.")

# Parámetros de la señal
lowcut = 100
highcut = 200
nyquist = 0.5 * fs

# Definir filtros IIR con diferentes configuraciones
iir_filters = {
    "Butterworth_4": butter(4, [lowcut / nyquist, highcut / nyquist], btype='band'),
    "Chebyshev1_4": cheby1(4, 0.2, [105 / nyquist, highcut / nyquist], btype='band'),
    "Chebyshev2_4": cheby2(4, 40, [70 / nyquist, 260 / nyquist], btype='band'),
    "Elliptic_4": ellip(4, 0.1, 50, [105 / nyquist, highcut / nyquist], btype='band'),
}

# Aplicar los filtros
filtered_signals = {}
for name, (b, a) in iir_filters.items():
    try:
        filtered_signal = filtfilt(b, a, signal_data)
        # Verificar si hay NaN o Inf después del filtrado
        if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
            print(f"Advertencia: {name} generó valores NaN o Inf. Se reemplazarán por ceros.")
            filtered_signal = np.nan_to_num(filtered_signal)  # Reemplazar NaN/Inf por 0
        filtered_signals[name] = filtered_signal
    except Exception as e:
        print(f"Error al aplicar {name}: {e}")
        filtered_signals[name] = np.zeros_like(signal_data)

# Evaluar métricas
metrics = []
for name, filtered_signal in filtered_signals.items():
    mse = mean_squared_error(signal_data, filtered_signal)
    snr = calculate_snr(filtered_signal, signal_data - filtered_signal)
    corr = calculate_pearson_correlation(filtered_signal, signal_data)
    distortion = calculate_signal_distortion(signal_data, filtered_signal)
    sdr = calculate_sdr(signal_data, filtered_signal)

    metrics.append([name, mse, snr, corr, distortion, sdr])

# Convertir a DataFrame y guardar en CSV
df = pd.DataFrame(metrics, columns=["Filtro", "MSE (Error Cuadrático Medio)", "SNR (Relación Señal-Ruido)",
                                    "Correlación de Pearson", "Signal Distortion", "SDR (Relación Señal-Distorsión)"])
csv_path = os.path.join(file_dir, "filtros_orden4.csv")
df.to_csv(csv_path, index=False)

print(f"Resultados guardados en {csv_path}")

# Imprimir resultados
for _, row in df.iterrows():
    print(f"--- {row['Filtro']} ---")
    print(f"MSE: {row['MSE (Error Cuadrático Medio)']:.4f}")
    print(f"SNR: {row['SNR (Relación Señal-Ruido)']:.4f}")
    print(f"Correlation: {row['Correlación de Pearson']:.4f}")
    print(f"Signal Distortion: {row['Signal Distortion']:.4f}")
    print(f"SDR: {row['SDR (Relación Señal-Distorsión)']:.4f}")
    print()

# Crear figura con subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Análisis de Filtros IIR", fontsize=16)

# Respuestas en frecuencia de los filtros
for name, (b, a) in iir_filters.items():
    w, h = freqz(b, a, worN=2000)
    axs[0, 0].plot(0.5 * fs * w / np.pi, np.abs(h), label=name)
axs[0, 0].set_title("Respuesta en Frecuencia")
axs[0, 0].set_xlabel("Frecuencia [Hz]")
axs[0, 0].set_ylabel("Ganancia")
axs[0, 0].set_xlim(0, 300)
axs[0, 0].legend()
axs[0, 0].grid()

# Respuesta al impulso
t = np.arange(50)
for name, (b, a) in iir_filters.items():
    system = dlti(b, a)
    _, imp = dimpulse(system, n=50)
    axs[0, 1].plot(t, np.squeeze(imp), label=name)
axs[0, 1].set_title("Respuesta al Impulso")
axs[0, 1].set_xlabel("Muestras")
axs[0, 1].set_ylabel("Amplitud")
axs[0, 1].legend()
axs[0, 1].grid()

# Comparación del espectro
frequencies = np.fft.fftfreq(len(signal_data), 1/fs)
signal_spectrum = np.abs(np.fft.fft(signal_data))
axs[1, 0].plot(frequencies[:len(frequencies)//2], signal_spectrum[:len(frequencies)//2], label="Original", alpha=0.1)
for name, filtered_signal in filtered_signals.items():
    filtered_spectrum = np.abs(np.fft.fft(filtered_signal))
    axs[1, 0].plot(frequencies[:len(frequencies)//2], filtered_spectrum[:len(frequencies)//2], label=name, alpha=0.7)
axs[1, 0].set_title("Espectro de la Señal Filtrada")
axs[1, 0].set_xlabel("Frecuencia (Hz)")
axs[1, 0].set_ylabel("Magnitud")
axs[1, 0].set_xlim(0, 300)
axs[1, 0].set_ylim(0, 40000)
axs[1, 0].legend()
axs[1, 0].grid()

# SNR por segmentos
segment_length = 1000
snr_values = {name: [] for name in filtered_signals}
for i in range(0, len(signal_data), segment_length):
    seg_signal = signal_data[i:i+segment_length]
    for name, filtered_signal in filtered_signals.items():
        seg_filtered = filtered_signal[i:i+segment_length]
        noise = seg_signal - seg_filtered
        snr_values[name].append(calculate_snr(seg_filtered, noise))
for name, snr in snr_values.items():
    axs[1, 1].plot(snr, label=name)
axs[1, 1].set_title("SNR por Segmentos")
axs[1, 1].set_xlabel("Segmento")
axs[1, 1].set_ylabel("SNR (dB)")
axs[1, 1].legend()
axs[1, 1].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
