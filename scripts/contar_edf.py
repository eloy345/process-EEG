import os
import pandas as pd

def contar_edf_y_guardar_lista(ruta_subcarpeta, etiqueta):
    rutas_relativas = []
    for root, _, files in os.walk(ruta_subcarpeta):
        for file in files:
            if file.lower().endswith('.edf'):
                ruta_rel = os.path.relpath(os.path.join(root, file), start=ruta_subcarpeta)
                rutas_relativas.append(os.path.join(etiqueta, ruta_rel).replace("\\", "/"))
    return rutas_relativas

# Rutas base
base_raw = "EEG_raw"
ruta_pre = os.path.join(base_raw, "PRE")
ruta_post = os.path.join(base_raw, "POST")

# Recoger rutas relativas
rutas_pre = contar_edf_y_guardar_lista(ruta_pre, "PRE")
rutas_post = contar_edf_y_guardar_lista(ruta_post, "POST")

# Contar
print(f"📂 Archivos .edf encontrados:")
print(f"  ▶ PRE:  {len(rutas_pre)}")
print(f"  ▶ POST: {len(rutas_post)}")
print(f"📊 Diferencia (POST - PRE): {len(rutas_post) - len(rutas_pre)}")

# Guardar en CSV
df = pd.DataFrame(rutas_pre + rutas_post, columns=["ruta_relativa"])
df.to_csv("archivos_edf_detectados.csv", index=False, encoding="utf-8")
print("✅ CSV guardado como 'archivos_edf_detectados.csv'")

import os
import pandas as pd

def recopilar_archivos(base_dir, nombre_archivo):
    """
    Devuelve las rutas relativas (sin nombre de archivo) donde se encuentra `nombre_archivo`.
    """
    rutas = set()
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file == nombre_archivo:
                rel = os.path.relpath(os.path.join(root, file), base_dir)
                ruta_sin_archivo = os.path.dirname(rel).replace("\\", "/")
                rutas.add(ruta_sin_archivo)
    return rutas

# Rutas base
ruta_raw = "EEG_crop"
ruta_crop = "EEG_processed_bandas"

# Buscar carpetas que contienen aacc.edf y aacc_basal.fif
carpetas_con_edf = recopilar_archivos(ruta_raw, "aacc_basal.fif")
carpetas_con_basal = recopilar_archivos(ruta_crop, "aacc_basal.fif")

# Comparar: ¿qué carpetas tienen aacc.edf pero no tienen aacc_basal.fif?
faltantes = sorted(carpetas_con_edf - carpetas_con_basal)

# Guardar resultado
df_faltantes = pd.DataFrame(faltantes, columns=["carpetas_sin_crop_basal"])
df_faltantes.to_csv("faltantes_crop_basal.csv", index=False, encoding="utf-8")
print("✅ Resultado guardado en 'faltantes_crop_basal.csv'")




# Ruta base donde buscar archivos que contengan 'aacc-proc'
ruta_crop = "EEG_crop"
archivos_borrados = []

# Recorrer directorios y eliminar archivos que contengan 'aacc-proc'
for raiz, _, archivos in os.walk(ruta_crop):
    for archivo in archivos:
        if 'aacc-proc' in archivo and archivo.endswith('.fif'):
            ruta_completa = os.path.join(raiz, archivo)
            try:
                os.remove(ruta_completa)
                archivos_borrados.append(os.path.relpath(ruta_completa, ruta_crop))
            except Exception as e:
                archivos_borrados.append(f"{ruta_completa} ❌ Error: {e}")

# Mostrar resultados
import pandas as pd


df_borrados = pd.DataFrame(archivos_borrados, columns=["archivos_eliminados"])
df_borrados.to_csv("archivos_eliminados.csv", index=False, encoding='utf-8')
print("✅ CSV de archivos eliminados guardado como 'archivos_eliminados.csv'")