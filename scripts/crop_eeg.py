
import os
import mne
import csv
import pandas as pd
from cargar_datos_raw import diccionario_datos

ruta_base = "EEG_raw"
salida_base = "EEG_crop"
rutas = diccionario_datos(ruta_base)
saltados = []
resumen_general = []

def buscar_fila_mas_cercana(df, tiempo_objetivo):
    return (df['timestamp'] - tiempo_objetivo).abs().idxmin()

for ruta_relativa, ruta_absoluta in rutas.items():
    print(f"\nProcesando: {ruta_relativa}")
    try:
        raw = mne.io.read_raw_edf(ruta_absoluta, preload=True)
        rename_dict = {ch: ch.replace('eeg_', '') for ch in raw.ch_names if ch.startswith('eeg_')}
        raw.rename_channels(rename_dict)

        canales_no_eeg = ['mask_eeg', 'mask_imu'] + [f'imu_{i}' for i in range(1, 10)]
        canales_en_raw = set(raw.ch_names)
        canales_validos = list(set(canales_no_eeg) & canales_en_raw)

        if canales_validos:
            raw.set_channel_types({canal: 'misc' for canal in canales_validos})

        raw.set_montage('standard_1020', on_missing='ignore')
    except Exception as e:
        print(f"❌ Error al cargar {ruta_relativa}: {e}")
        saltados.append((ruta_relativa, f"Error al abrir: {e}"))
        continue

    anotaciones = raw.annotations
    descripciones = list(anotaciones.description)

    if "userMarker_5" in descripciones:
        print(f"❌ Saltado (tiene userMarker_5): {ruta_relativa}")
        saltados.append((ruta_relativa, "Contiene userMarker_5"))
        continue

    try:
        t1 = anotaciones.onset[descripciones.index("userMarker_1")]
        t2 = anotaciones.onset[descripciones.index("userMarker_2")]
        t3 = anotaciones.onset[descripciones.index("userMarker_3")]

        t4_propuesto = t3 + 420.0
        duracion_total = raw.times[-1]

        if t4_propuesto <= duracion_total:
            t4 = t4_propuesto
        else:
            if "userMarker_4" in descripciones:
                t4 = anotaciones.onset[descripciones.index("userMarker_4")]
                print(f"⚠️ {ruta_relativa} - 7 minutos no disponibles, usando userMarker_4 como t4")
            else:
                print(f"❌ {ruta_relativa} - No hay suficientes datos ni userMarker_4, se omite")
                saltados.append((ruta_relativa, "No hay t4 válido (ni 7 min ni userMarker_4)"))
                continue
    except ValueError as e:
        print(f"❌ Faltan marcadores clave en: {ruta_relativa} → {e}")
        saltados.append((ruta_relativa, str(e)))
        continue

    raw_basal = raw.copy().crop(tmin=t1, tmax=t2)
    raw_prueba = raw.copy().crop(tmin=t3, tmax=t4)

    ruta_quality = os.path.join(os.path.dirname(ruta_absoluta), "AACC_QUALITY.csv")
    ruta_usermarker = os.path.join(os.path.dirname(ruta_absoluta), "userMarker.csv")

    canales_malos_basal = []
    canales_malos_vb = []

    if os.path.exists(ruta_quality) and os.path.exists(ruta_usermarker):
        try:
            df_quality = pd.read_csv(ruta_quality)
            df_marker = pd.read_csv(ruta_usermarker)

            ts1 = int(df_marker[df_marker["value1"] == 1]["timestamp"].values[0])
            ts2 = int(df_marker[df_marker["value1"] == 2]["timestamp"].values[0])
            ts3 = int(df_marker[df_marker["value1"] == 3]["timestamp"].values[0])
            ts4 = int(df_marker[df_marker["value1"] == 4]["timestamp"].values[0]) if 4 in df_marker["value1"].values else int(ts3 + 420 * 1e6)

            i1 = buscar_fila_mas_cercana(df_quality, ts1)
            i2 = buscar_fila_mas_cercana(df_quality, ts2)
            i3 = buscar_fila_mas_cercana(df_quality, ts3)
            i4 = buscar_fila_mas_cercana(df_quality, ts4)

            df_basal = df_quality.iloc[i1:i2]
            df_vb = df_quality.iloc[i3:i4]

            canales = raw_basal.ch_names
            resumen = []

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

                resumen.append({
                    "archivo": ruta_relativa,
                    "canal": canal,
                    "rojo_basal_%": round(pct_rojo_basal * 100, 2),
                    "bad_basal": pct_rojo_basal > 0.3,
                    "rojo_vb_%": round(pct_rojo_vb * 100, 2),
                    "bad_vb": pct_rojo_vb > 0.3,
                })

            raw_basal.info["bads"] = canales_malos_basal
            raw_prueba.info["bads"] = canales_malos_vb

            resumen_general.extend(resumen)

        except Exception as e:
            print(f"⚠️ Error analizando calidad: {e}")
    else:
        print("⚠️ No se encontró el archivo AACC_QUALITY.csv o userMarker.csv")

    ruta_salida_basal = os.path.join(salida_base, os.path.dirname(ruta_relativa), "aacc_basal.fif")
    ruta_salida_vb = os.path.join(salida_base, os.path.dirname(ruta_relativa), "aacc_vb.fif")

    os.makedirs(os.path.dirname(ruta_salida_basal), exist_ok=True)
    raw_basal.save(ruta_salida_basal, overwrite=True)
    raw_prueba.save(ruta_salida_vb, overwrite=True)

    print(f"✅ Guardado: {ruta_salida_basal} y {ruta_salida_vb}")

with open("archivos_saltados.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Ruta relativa", "Motivo"])
    writer.writerows(saltados)

# Guardar resumen de calidad total
pd.DataFrame(resumen_general).to_csv("resumen_malos.csv", index=False)
print("\n📄 Archivos 'archivos_saltados.csv' y 'resumen_malos.csv' generados.")
