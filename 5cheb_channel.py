import os
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
from scipy.signal import butter, cheby2, filtfilt, hilbert

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

# Select channel
selected_channel = 24
signal_data = signal_data[selected_channel]
ch_name = f"ch_{selected_channel + 1}"

lowcut = 100
highcut = 200
nyquist = 0.5 * fs

# Filter signal
b, a = cheby2(4, 40, [70 / nyquist, 260 / nyquist], btype='band')
data_np = cp.asnumpy(signal_data)
filtered_data = filtfilt(b, a, data_np)

# Compute envelope
envelope = cp.abs(cp.array(hilbert(filtered_data)))

# Detect ripples
ripple_events = []
refractory_time_sec = 0.015  # Período refractario en segundos
refractory_period = int(refractory_time_sec * fs)


# Threshold: 75th percentile + 3 std deviations
threshold = cp.percentile(envelope, 75) + 3 * cp.std(envelope)
ripple_mask = envelope > threshold
ripple_indices = cp.where(ripple_mask)[0]

# Filter close events
filtered_ripple_indices = []
last_event = -refractory_period
for idx in ripple_indices:
    if idx > last_event + refractory_period:
        filtered_ripple_indices.append(idx)
        last_event = idx

ripple_times = cp.array(filtered_ripple_indices) / fs
if len(ripple_times) > 0:
    ripple_events.append(pd.DataFrame({'file': filename, 'channel': ch_name, 'time': cp.asnumpy(ripple_times)}))

# Save ripple events
event_df = pd.concat(ripple_events, ignore_index=True) if ripple_events else pd.DataFrame(
    columns=['file', 'channel', 'time'])
csv_filename = os.path.join(file_dir, f"ripple_events_{filename}.csv")
event_df.to_csv(csv_filename, index=False)
print(f"Eventos detectados guardados en 'ripple_events_{filename}.csv'.")

# Prepare data for plotting
signal_df = pd.DataFrame({
    'time': np.arange(len(signal_data)) / fs,
    'original': cp.asnumpy(signal_data),
    'filtered': cp.asnumpy(filtered_data),
    'threshold': cp.asnumpy(threshold)
})

# Animal movement windows
movement_windows = [(47, 50), (131, 135), (220, 224), (252, 254)]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

ylim_range = [(-600, 600), (-400, 400), (-200, 200), (-200, 200)]
xlim_range = [(0, signal_df['time'].max()), (70, 80), (75, 76), (75.1, 75.3)]

for i, ax in enumerate(axes):
    ax.plot(signal_df['time'], signal_df['original'], color='black', alpha=0.6, label='Señal Original')
    ax.plot(signal_df['time'], signal_df['filtered'], color='red', alpha=0.8, label='Señal Filtrada')
    ax.plot(signal_df['time'], signal_df['threshold'], color='blue', alpha=0.8, label='Threshold')

    # Show Ripples
    ripple_ch_events = event_df[event_df['channel'] == ch_name]
    for _, event in ripple_ch_events.iterrows():
        ax.axvspan(xmin=event['time'], xmax=event['time'] + 0.015, color='cyan', alpha=0.3)

    # Show movement
    for start, end in movement_windows:
        ax.axvspan(xmin=start, xmax=end, color='yellow', alpha=0.3)

    ax.set_xlim(xlim_range[i])
    ax.set_ylim(ylim_range[i])
    ax.set_title(f"Canal: {ch_name}")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Amplitud")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

