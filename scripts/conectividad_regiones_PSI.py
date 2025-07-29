
import os
import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from mne import make_fixed_length_epochs

# === CONFIG ===
DATA_DIR = "eeg_processed_bandas/PRE"
OUTPUT_DIR = "EEG_connectivity_PSI"
BAND = (4, 30)  # PSI se recomienda usarlo en un rango amplio
REGIONES = {
    "frontal": ["Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8"],
    "central": ["FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8"],
    "parietal": ["CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8"],
    "occipital": ["POz", "O1", "Oz", "O2"],
    "izquierda": ["Fp1", "AF3", "F7", "F3", "FC5", "FC1", "T7", "C3", "CP5", "CP1", "P7", "P3", "O1"],
    "derecha": ["Fp2", "AF4", "F4", "F8", "FC2", "FC6", "C4", "T8", "CP2", "CP6", "P4", "P8", "O2"]
}

def reducir_conectividad_a_regiones(con, ch_names, regiones):
    n = len(regiones)
    matriz = np.zeros((n, n))
    region_names = list(regiones.keys())
    name_to_idx = {ch: idx for idx, ch in enumerate(ch_names)}
    for i, r1 in enumerate(region_names):
        for j, r2 in enumerate(region_names):
            pares = [(name_to_idx[a], name_to_idx[b]) for a in regiones[r1] for b in regiones[r2]
                     if a in name_to_idx and b in name_to_idx]
            valores = [con[i1, i2] for i1, i2 in pares]
            matriz[i, j] = np.nanmean(valores) if valores else np.nan
    return region_names, matriz

# === MAIN ===
for root, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".fif"):
            filepath = os.path.join(root, file)
            print(f"📄 Procesando: {filepath}")
            raw = mne.io.read_raw_fif(filepath, preload=True)
            raw.pick("eeg")
            epochs = make_fixed_length_epochs(raw, duration=2.0, preload=True)

            con = spectral_connectivity_epochs(
                epochs, method="psi", sfreq=raw.info["sfreq"],
                fmin=BAND[0], fmax=BAND[1], faverage=True, verbose=False
            )

            matrix = con.get_data(output="dense")[:, :, 0]
            ch_names = raw.info["ch_names"]
            labels, region_matrix = reducir_conectividad_a_regiones(matrix[:, :, 0], ch_names, REGIONES)

            # Guardar
            rel = os.path.relpath(filepath, DATA_DIR)
            out_dir = os.path.join(OUTPUT_DIR, os.path.dirname(rel))
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(file))[0]
            out_path = os.path.join(out_dir, f"{base}-psi.npy")
            np.save(out_path, region_matrix)
            print(f"✅ Guardado: {out_path}")
