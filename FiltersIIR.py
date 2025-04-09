import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, cheby1, cheby2, ellip, sosfiltfilt, freqz_sos
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
dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection"
file_dir = os.path.join(dir, "processed_data")
data_dir = os.path.join(dir, "data")
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
output = 'sos'

# Definir filtros IIR con diferentes configuraciones
iir_filters = {
    "Butterworth_4": butter(8, [lowcut / nyquist, highcut / nyquist], btype='band', output=output),
    #"Butterworth_2": butter(2, [lowcut / nyquist, highcut / nyquist], btype='band', output=output),
    "Chebyshev1_4": cheby1(4, 0.3, [lowcut / nyquist, highcut / nyquist], btype='band', output=output),
    #"Chebyshev1_2": cheby1(2, 0.6, [110 / nyquist, 180 / nyquist], btype='band', output=output),
    "Chebyshev2_4": cheby2(4, 40, [70 / nyquist, 260 / nyquist], btype='band', output=output),
    #"Chebyshev2_2": cheby2(2, 30, [50 / nyquist, 300 / nyquist], btype='band', output=output),
    "Elliptic_4": ellip(4, 0.1, 40, [lowcut / nyquist, highcut / nyquist], btype='band', output=output),
    #"Elliptic_2": ellip(2, 0.1, 50, [110 / nyquist, 170 / nyquist], btype='band', output=output),
}

# Aplicar los filtros
filtered_signals = {}
for name, sos in iir_filters.items():
    try:
        filtered_signal = sosfiltfilt(sos, signal_data)
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
csv_filename = os.path.join(file_dir, "processed_data/filtros_resultados.csv")
df.to_csv(csv_filename, index=False)

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

# Graficar respuestas en frecuencia de los filtros
plt.figure(figsize=(10, 7))

for name, sos in iir_filters.items():
    w, h = freqz_sos(sos, worN=2000)
    plt.plot(0.5 * fs * w / np.pi, np.abs(h), label=name)

plt.title("Respuestas en Frecuencia de los Filtros IIR")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Ganancia")
plt.xlim([0, 300])
plt.legend(loc='best')
plt.grid(True)
plt.show()
