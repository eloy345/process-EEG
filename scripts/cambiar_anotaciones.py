#
import mne
import os
import pandas as pd

# ==== CONFIGURACIÓN MANUAL (MODIFICABLE POR EL USUARIO) ====
ruta_edf = "C:/Users/Eloy/OneDrive - Universidad de Castilla-La Mancha (1)/Tesis_EEG/proyecto_eeg/EEG_raw/PRE/KAIZEN/5/EEG/KAI/Session01/RecordingS01R000_EDF/aacc.edf"
ruta_base_raw = "EEG_raw"
ruta_base_proc = "EEG_crop"
marker_basal_inicio = 1
marker_basal_final = 2
marker_vb_inicio = 3
# ============================================================

# Cargar EDF
raw = mne.io.read_raw_edf(ruta_edf, preload=True)
rename_dict = {ch: ch.replace('eeg_', '') for ch in raw.ch_names if ch.startswith('eeg_')}
raw.rename_channels(rename_dict)

canales_no_eeg = ['mask_eeg', 'mask_imu'] + [f'imu_{i}' for i in range(1, 10)]
canales_validos = list(set(canales_no_eeg) & set(raw.ch_names))
if canales_validos:
    raw.set_channel_types({ch: 'misc' for ch in canales_validos})

raw.set_montage('standard_1020', on_missing='ignore')
anotaciones = raw.annotations
descripciones = list(anotaciones.description)

try:
    t_basal_ini = anotaciones.onset[descripciones.index(f"userMarker_{marker_basal_inicio}")]
    t_basal_fin = anotaciones.onset[descripciones.index(f"userMarker_{marker_basal_final}")]
    t_vb_ini = anotaciones.onset[descripciones.index(f"userMarker_{marker_vb_inicio}")]
    t_vb_fin = t_vb_ini + 420
except ValueError as e:
    print(f"❌ Error: {e}")
    exit()

raw_basal = raw.copy().crop(tmin=t_basal_ini, tmax=t_basal_fin)
raw_vb = raw.copy().crop(tmin=t_vb_ini, tmax=t_vb_fin)

# === Calidad de señal ===
ruta_quality = os.path.join(os.path.dirname(ruta_edf), "AACC_QUALITY.csv")
ruta_usermarker = os.path.join(os.path.dirname(ruta_edf), "userMarker.csv")

canales_malos_basal = []
canales_malos_vb = []
resumen = []

if os.path.exists(ruta_quality) and os.path.exists(ruta_usermarker):
    df_quality = pd.read_csv(ruta_quality)
    df_marker = pd.read_csv(ruta_usermarker)

    def get_timestamp(valor):
        fila = df_marker[df_marker["value1"] == valor]
        return int(fila["timestamp"].values[0]) if not fila.empty else None

    ts_basal_ini = get_timestamp(marker_basal_inicio)
    ts_basal_fin = get_timestamp(marker_basal_final)
    ts_vb_ini = get_timestamp(marker_vb_inicio)
    ts_vb_fin = int(ts_vb_ini + 420 * 1e6)

    def encontrar_fila(df, ts): return (df["timestamp"] - ts).abs().idxmin()

    i1, i2 = encontrar_fila(df_quality, ts_basal_ini), encontrar_fila(df_quality, ts_basal_fin)
    i3, i4 = encontrar_fila(df_quality, ts_vb_ini), encontrar_fila(df_quality, ts_vb_fin)

    df_basal = df_quality.iloc[i1:i2]
    df_vb = df_quality.iloc[i3:i4]
    canales = raw_basal.ch_names

    for i, canal in enumerate(canales):
        col = f"value{i+1}"
        if col not in df_quality.columns:
            continue
        datos_basal = df_basal[col].dropna()
        datos_vb = df_vb[col].dropna()

        pct_rojo_basal = (datos_basal == 3).sum() / len(datos_basal) if len(datos_basal) else 0
        pct_rojo_vb = (datos_vb == 3).sum() / len(datos_vb) if len(datos_vb) else 0

        if pct_rojo_basal > 0.3:
            canales_malos_basal.append(canal)
        if pct_rojo_vb > 0.3:
            canales_malos_vb.append(canal)

        if pct_rojo_basal > 0.3 or pct_rojo_vb > 0.3:
            resumen.append({
                "canal": canal,
                "rojo_basal_%": round(pct_rojo_basal * 100, 2),
                "bad_basal": pct_rojo_basal > 0.3,
                "rojo_vb_%": round(pct_rojo_vb * 100, 2),
                "bad_vb": pct_rojo_vb > 0.3
            })

    raw_basal.info["bads"] = canales_malos_basal
    raw_vb.info["bads"] = canales_malos_vb

    # Guardar CSV de resumen
    df_resumen = pd.DataFrame(resumen)
    if not df_resumen.empty:
        resumen_path = os.path.join(ruta_base_proc, os.path.dirname(os.path.relpath(ruta_edf, ruta_base_raw)), "resumen_malos.csv")
        os.makedirs(os.path.dirname(resumen_path), exist_ok=True)
        df_resumen.to_csv(resumen_path, index=False)
        print("✅ Canal malos guardados en resumen_malos.csv")

# Guardado final 
rel_path = os.path.relpath(os.path.dirname(ruta_edf), ruta_base_raw)
ruta_salida_basal = os.path.join(ruta_base_proc, rel_path, "aacc_basal.fif")
ruta_salida_vb = os.path.join(ruta_base_proc, rel_path, "aacc_vb.fif")
os.makedirs(os.path.dirname(ruta_salida_basal), exist_ok=True)
raw_basal.save(ruta_salida_basal, overwrite=True)
raw_vb.save(ruta_salida_vb, overwrite=True)
print(f"✅ Guardado:\n- {ruta_salida_basal}\n- {ruta_salida_vb}")
