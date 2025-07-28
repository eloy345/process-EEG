
import os
import numpy as np
import pandas as pd
import mne
from scipy import signal
from tqdm import tqdm
from pathlib import Path

# 1. Extraer características
def extraer_caracteristicas(df, sfreq=256):
    canales_esperados = [
        "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8",
        "POz", "O1", "Oz", "O2"
    ]

    canales_presentes = [ch for ch in canales_esperados if ch in df.columns]
    df = df[canales_presentes]
    D1 = df.diff().fillna(0)
    D2 = D1.diff().fillna(0)
    media, std = df.mean(), df.std()
    maximo, minimo = df.max(), df.min()
    RD = maximo - minimo
    MD1, STDD1 = D1.mean(), D1.std()
    MD2, STDD2 = D2.mean(), D2.std()
    AL, RM, INT = [], [], []
    for ch in canales_presentes:
        x = df[ch].values
        al = np.sum(np.sqrt(1 + np.diff(x)**2)) / len(x)
        ap = np.sum(np.abs(x)**2) / len(x)
        rm = np.sqrt(ap)
        integ = np.sum(np.abs(x)) / len(x)
        AL.append(al)
        RM.append(rm)
        INT.append(integ)
    AL = pd.Series(AL, index=canales_presentes)
    RM = pd.Series(RM, index=canales_presentes)
    INT = pd.Series(INT, index=canales_presentes)
    IL = media / AL
    EL = RM / AL
    skewness = df.skew()
    kurtosis = df.kurt()

    potencia_abs = {}
    for ch in canales_presentes:
        freqs, psd = signal.welch(df[ch].values, fs=sfreq, nperseg=sfreq*2)
        potencia_abs[ch] = np.trapezoid(psd, freqs)
    potencia_abs = pd.Series(potencia_abs)
    potencia_rel = potencia_abs / potencia_abs.sum()

    return pd.concat([
        media.add_suffix('_mean'),
        std.add_suffix('_std'),
        maximo.add_suffix('_max'),
        minimo.add_suffix('_min'),
        RD.add_suffix('_range'),
        MD1.add_suffix('_d1_mean'),
        STDD1.add_suffix('_d1_std'),
        MD2.add_suffix('_d2_mean'),
        STDD2.add_suffix('_d2_std'),
        AL.rename(lambda ch: f"{ch}_arc_len"),
        RM.rename(lambda ch: f"{ch}_rms"),
        IL.rename(lambda ch: f"{ch}_area_perim"),
        EL.rename(lambda ch: f"{ch}_energy_perim"),
        INT.rename(lambda ch: f"{ch}_integral"),
        skewness.rename(lambda ch: f"{ch}_skew"),
        kurtosis.rename(lambda ch: f"{ch}_kurt"),
        potencia_abs.rename(lambda ch: f"{ch}_welch_abs"),
        potencia_rel.rename(lambda ch: f"{ch}_welch_rel")
    ])

# 2. Filtrado por banda
def cargar_y_filtrar(path_fif, l_freq, h_freq):
    raw = mne.io.read_raw_fif(path_fif, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', verbose=False)
    data, _ = raw[:]
    return pd.DataFrame(data.T, columns=raw.ch_names)

# 3. Dividir en ventanas
def dividir_en_ventanas(df, sfreq=256, ventana_s=10, solape=0):
    ventana_muestras = int(sfreq * ventana_s)
    paso = int(ventana_muestras * (1 - solape))
    ventanas = []
    for inicio in range(0, len(df) - ventana_muestras + 1, paso):
        fin = inicio + ventana_muestras
        ventanas.append(df.iloc[inicio:fin].reset_index(drop=True))
    return ventanas

# 4. Configuración
bandas = {
    "full": (1, 40),
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40)
}

ruta_base = "EEG_processed_bandas/PRE"
ruta_excel = "Experimento_EEG.xlsx"
ruta_guardado = "df_EEG_windowed/PRE"
Path(ruta_guardado).mkdir(parents=True, exist_ok=True)

# 5. Metadatos
df_excel = pd.read_excel(ruta_excel)
df_excel["ID"] = df_excel["INICIALES"].astype(str) + "_" + df_excel["CENTRO_ESCOLAR"].astype(str) + "_" + df_excel["NIVEL_EDUCATIVO"].astype(str)
df_excel = df_excel.set_index("ID")
df_resultados = {banda: [] for banda in bandas}
meta_info = []

# 6. Procesamiento principal
for carpeta in tqdm(sorted(os.listdir(ruta_base))):
    ruta_carpeta = os.path.join(ruta_base, carpeta)
    if not os.path.isdir(ruta_carpeta):
        continue

    for tipo, ventana_s in zip(["aacc_basal.fif", "aacc_vb.fif"], [90, 180  ]):
        ruta_fif = os.path.join(ruta_carpeta, tipo)
        if not os.path.exists(ruta_fif):
            continue
        for nombre_banda, (lf, hf) in bandas.items():
            try:
                df_raw = cargar_y_filtrar(ruta_fif, lf, hf)
                ventanas = dividir_en_ventanas(df_raw, ventana_s=ventana_s, solape=0)
                for i, df_v in enumerate(ventanas):
                    features = extraer_caracteristicas(df_v)
                    features["ID"] = carpeta
                    features["ventana"] = i
                    features["tipo"] = tipo.replace(".fif", "")
                    clave = f"{nombre_banda}_{tipo.replace('.fif', '')}"
                    df_resultados.setdefault(clave, []).append(features)
            except Exception as e:
                print(f"❌ Error en {carpeta}/{tipo} - {nombre_banda}: {e}")

    if carpeta in df_excel.index:
        info = df_excel.loc[carpeta].to_dict()
        info["ID"] = carpeta
        meta_info.append(info)



# 7. Guardado
for clave, lista_df in df_resultados.items():
    df_final = pd.DataFrame(lista_df)
    df_final = df_final.reset_index(drop=True)
    columnas_presentes = df_final.columns.tolist()
    columnas_base = [col for col in ['ID', 'ventana', 'tipo'] if col in columnas_presentes]
    columnas_extra = [col for col in columnas_presentes if col not in columnas_base]
    columnas_ordenadas = columnas_base + columnas_extra
    df_final = df_final[columnas_ordenadas]
    df_final.to_csv(f"{ruta_guardado}/X_{clave}.csv", index=False)

df_meta = pd.DataFrame(meta_info).set_index("ID")
df_meta["y"] = df_meta["AACC"].apply(lambda x: 1 if str(x) == "SI" else 0)
df_meta["y_recommended"] = df_meta["AACC"].apply(lambda x: 1 if pd.notna(x) and str(x).strip() != "" else 0)
df_meta.to_csv(f"{ruta_guardado}/meta.csv")
