import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from sklearn.preprocessing import StandardScaler

# ---------- RUTAS ----------
ruta_X = "df_EEG/PRE/X_gamma.csv"
ruta_meta = "df_EEG/PRE/meta.csv"

# ---------- CARGA ----------
df_X = pd.read_csv(ruta_X, index_col=0)
df_meta = pd.read_csv(ruta_meta, index_col=0)
df = df_X.join(df_meta[["y_recommended"]], how="inner")

# ---------- SELECCIÓN DE CANALES REALES ----------
canales = [col for col in df.columns if col.endswith("_mean") and "_d1_" not in col and "_d2_" not in col]
nombres_canales = [col.replace("_mean", "") for col in canales]

# ---------- NORMALIZACIÓN GLOBAL ----------
scaler = StandardScaler()
df[canales] = scaler.fit_transform(df[canales])

# ---------- AGRUPACIÓN ----------
df_aacc = df[df["y_recommended"] == 1]
df_noaacc = df[df["y_recommended"] == 0]
medias_aacc = df_aacc[canales].mean()
medias_noaacc = df_noaacc[canales].mean()

# ---------- MONTAJE TOPOGRÁFICO ----------
info = mne.create_info(ch_names=nombres_canales, sfreq=128, ch_types="eeg")
montaje = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montaje)
pos_dict = info.get_montage().get_positions()['ch_pos']
pos = np.array([pos_dict[ch][:2] for ch in nombres_canales])

# ---------- FUNCIÓN TOPOPLOT ----------
def plot_topoplot(valores, titulo, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im, _ = mne.viz.plot_topomap(
        valores.values, pos, axes=ax, sensors=True,
        cmap="viridis", contours=0, sphere=0.095,
        image_interp='cubic', show=False
    )
    im.set_clim(vmin, vmax)  # ✅ Ajusta límites manualmente
    
    for i, name in enumerate(nombres_canales):
        x, y = pos[i]
        ax.text(x, y + 0.001, name, fontsize=13, ha='center', va='center', color='black')

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', shrink=0.7)
    cbar.set_label("Mean by Channel (Normalized)", fontsize=14)
    cbar.ax.tick_params(labelsize=16) 
    ax.set_title(titulo, fontsize=18)
    plt.tight_layout()
    plt.show()

# ---------- MOSTRAR TOPOPLOTS ----------
plot_topoplot(medias_aacc, "Gamma - AACC", vmin=-0.3, vmax=0.3)
plot_topoplot(medias_noaacc, "Gamma - No AACC", vmin=-0.3, vmax=0.3)
