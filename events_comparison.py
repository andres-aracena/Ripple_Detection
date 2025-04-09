import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths to CSVs
csv_normal = "processed_data/ripples_datafile002.ns6.csv"
csv_gauss = "processed_data/ripples2_datafile002.ns6.csv"

# Load data
df_normal = pd.read_csv(csv_normal)
df_gauss = pd.read_csv(csv_gauss)

# Ensure 'start' is float
df_normal['start'] = df_normal['start'].astype(float)
df_gauss['start'] = df_gauss['start'].astype(float)

# Tolerance (s)
tolerance = 0.005

# Sort for efficient search
df_normal_sorted = df_normal.sort_values(by='start').reset_index(drop=True)
df_gauss_sorted = df_gauss.sort_values(by='start').reset_index(drop=True)

# Convert to numpy for speed
starts_normal = df_normal_sorted['start'].values
starts_gauss = df_gauss_sorted['start'].values

# Unique in NORMAL
mask_normal = np.array([
    not np.any(np.abs(starts_gauss - s) <= tolerance) for s in starts_normal
])
df_unique_normal = df_normal_sorted[mask_normal].copy()

# Unique in GAUSS
mask_gauss = np.array([
    not np.any(np.abs(starts_normal - s) <= tolerance) for s in starts_gauss
])
df_unique_gauss = df_gauss_sorted[mask_gauss].copy()

# Add source
df_unique_normal['source'] = 'Normal'
df_unique_gauss['source'] = 'Gauss'

# Combine and save
df_combined = pd.concat([df_unique_normal, df_unique_gauss], ignore_index=True)
df_combined.to_csv("processed_data/unique_events2.csv", index=False)

# Report
if df_combined.empty:
    print("⚠️ No hay eventos únicos para guardar.")
else:
    print("✅ CSV guardado con eventos únicos.")

print("Unique in NORMAL:", len(df_unique_normal))
print(df_unique_normal[['start', 'end', 'duration']] if not df_unique_normal.empty else "Ninguno.")

print("\nUnique in GAUSS:", len(df_unique_gauss))
print(df_unique_gauss[['start', 'end', 'duration']] if not df_unique_gauss.empty else "Ninguno.")

# Zoom range
t_min = 60
t_max = 120

# Plot
fig, axs = plt.subplots(2, 1, figsize=(15, 10))

axs[0].barh(
    y=[1]*len(df_normal),
    width=df_normal['duration']*3,
    left=df_normal['start'],
    height=0.3,
    color='blue',
    alpha=0.7,
    label='Normal'
)
axs[0].barh(
    y=[0]*len(df_gauss),
    width=df_gauss['duration']*3,
    left=df_gauss['start'],
    height=0.3,
    color='green',
    alpha=0.7,
    label='Gauss'
)
axs[0].set_yticks([0, 1])
axs[0].set_yticklabels(['Gauss', 'Normal'])

axs[0].set_xlabel("Time (s)")
axs[0].set_title("Event comparison")

axs[1].barh(
    y=[1]*len(df_normal),
    width=df_normal['duration'],
    left=df_normal['start'],
    height=0.3,
    color='blue',
    alpha=0.6,
    label='Normal'
)
axs[1].barh(
    y=[0]*len(df_gauss),
    width=df_gauss['duration'],
    left=df_gauss['start'],
    height=0.3,
    color='green',
    alpha=0.6,
    label='Gauss'
)
axs[1].set_yticks([0, 1])
axs[1].set_yticklabels(['Gauss', 'Normal'])
axs[1].set_xlabel("Time (s)")
axs[1].set_title("Event comparison (zoomed)")
axs[1].set_xlim(t_min, t_max)

plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()
