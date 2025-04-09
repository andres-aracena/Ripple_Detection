import os
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neo
from scipy.signal import butter, cheby2, filtfilt, hilbert

# Set directory and file
dir = "C:/Users/Andres/OneDrive/Documentos/Anaconda/Ripple_Detection"
file_dir = os.path.join(dir, "processed_data")
data_dir = os.path.join(dir, "data")
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

# Threshold: 75th percentile + 3 std deviations
threshold = cp.percentile(envelope, 75) + 3 * cp.std(envelope)
ripple_mask = envelope > threshold

# Parameters for validation
min_ripple_duration_sec = 0.015  # Minimum 15 ms for validation ripple
min_ripple_samples = int(min_ripple_duration_sec * fs)
max_gap_sec = 0.005  # Gap máximo permitido
max_gap_samples = int(max_gap_sec * fs)
post_ripple_gap_sec = 0.010  # Previous events windows
post_ripple_gap_samples = int(post_ripple_gap_sec * fs)

# Detección optimizada con llenado de gaps en tiempo real
ripple_events = []
start_idx = None
end_idx = None
last_start = 0
last_end = 0
gap = 0

for i in range(len(ripple_mask)):
    if ripple_mask[i]:
        if start_idx is None:
            start_idx = i
        end_idx = i
    else:
        if start_idx is not None:
            duration = end_idx - start_idx + 1
            gap = start_idx - last_end

            if gap <= max_gap_samples and (end_idx - last_start) >= min_ripple_samples:
                ripple_events.append((last_start, end_idx))
            elif duration >= min_ripple_samples:
                ripple_events.append((start_idx, end_idx))
            # if duration >= min_ripple_samples:
            #     ripple_events.append((start_idx, end_idx))
            # elif ripple_events:
            #     last_start, last_end = ripple_events[-1]
            #     gap = start_idx - last_end
            #     if gap <= max_gap_samples and (end_idx - last_start) >= min_ripple_samples:
            #         ripple_events[-1] = (last_start, end_idx)

            last_start = start_idx
            last_end = end_idx
            start_idx = None


ripple_starts, ripple_ends = zip(*ripple_events) if ripple_events else ([], [])
ripple_starts = np.array(ripple_starts) / fs
ripple_ends = np.array(ripple_ends) / fs
ripple_durations = (ripple_ends - ripple_starts)


event_df = pd.DataFrame({
    'file': filename,
    'channel': ch_name,
    'duration': ripple_durations,
    'start': ripple_starts,
    'end': ripple_ends
})

# Guardar en CSV
csv_filename = os.path.join(file_dir, f"ripples_{filename}.csv")
event_df.to_csv(csv_filename, index=False)
print(f"Eventos detectados guardados en 'ripples_{filename}.csv'.")

# Convert data to numpy for plotting
time_array = np.arange(len(signal_data)) / fs
envelope_np = cp.asnumpy(envelope)

# Compute rolling standard deviation
rolling_std = pd.Series(envelope_np).rolling(window=post_ripple_gap_samples, center=True).std().fillna(0)

# Prepare data for plotting
signal_df = pd.DataFrame({
    'time': time_array,
    #'original': cp.asnumpy(signal_data),
    'filtered': cp.asnumpy(filtered_data),
    'threshold': cp.asnumpy(threshold),
    'envelope': envelope_np,
    'deviation': rolling_std
})

# Animal movement windows
#movement_windows = [(47, 50), (131, 135), (220, 224), (252, 254)]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()

ylim_range = [(-200, 200), (-100, 100), (-100, 100), (-100, 100)]
xlim_range = [(0, signal_df['time'].max()), (70, 80), (74, 75), (74.5, 74.89)]

for i, ax in enumerate(axes):
    #ax.plot(signal_df['time'], signal_df['original'], color='black', alpha=0.6, label='Señal Original')
    ax.plot(signal_df['time'], signal_df['filtered'], color='red', alpha=0.8, label='Señal Filtrada')
    ax.plot(signal_df['time'], signal_df['threshold'], color='blue', alpha=0.8, label='Threshold')
    ax.plot(signal_df['time'], signal_df['envelope'], color='green', alpha=0.8, label='Envelope')

    # Show STD
    ax.fill_between(signal_df['time'],
                     signal_df['envelope'] - signal_df['deviation'],
                     signal_df['envelope'] + signal_df['deviation'],
                     alpha=0.2, color="green", label="Desviación estándar")

    # Show Ripples
    ripple_ch_events = event_df[event_df['channel'] == ch_name]
    for _, event in ripple_ch_events.iterrows():
        ax.axvspan(xmin=event['start'], xmax=event['end'], color='cyan', alpha=0.3)

    # Show movement
    #for start, end in movement_windows:
    #    ax.axvspan(xmin=start, xmax=end, color='yellow', alpha=0.3)

    ax.set_xlim(xlim_range[i])
    ax.set_ylim(-100,100)
    ax.set_title(f"Canal: {ch_name}")
    ax.set_xlabel("Tiempo (segundos)")
    ax.set_ylabel("Amplitud")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

