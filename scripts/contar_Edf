# Compara si todos los edf están en fif

import os

def contar_edf(ruta_base):
    edf_files = []
    for root, _, files in os.walk(ruta_base):
        for f in files:
            if f.lower().endswith(".edf"):
                edf_files.append(os.path.relpath(os.path.join(root, f), ruta_base))
    return edf_files

def contar_pares_fif(ruta_base):
    pares_validos = []
    rutas_encontradas = set()
    for root, _, files in os.walk(ruta_base):
        base = os.path.relpath(root, ruta_base)
        tiene_basal = "aacc_basal.fif" in files
        tiene_vb = "aacc_vb.fif" in files
        if tiene_basal and tiene_vb:
            pares_validos.append(base)
        elif tiene_basal or tiene_vb:
            rutas_encontradas.add(base)
    return pares_validos, rutas_encontradas

# Rutas base
raw_dir = "EEG_raw"
proc_dir = "EEG_processed"

# Recuento
edf_archivos = contar_edf(raw_dir)
pares_fif, parciales_fif = contar_pares_fif(proc_dir)

print("🔍 Recuento de archivos:")
print(f"📁 Archivos .edf en {raw_dir}: {len(edf_archivos)}")
print(f"✅ Pares completos (basal + vb) en {proc_dir}: {len(pares_fif)}")
print(f"⚠️ Carpetas con solo un .fif (incompletas): {len(parciales_fif - set(pares_fif))}")

# Mostrar diferencias
if len(edf_archivos) != len(pares_fif):
    print("\n📌 Diferencias detectadas:")
    faltan = len(edf_archivos) - len(pares_fif)
    if faltan > 0:
        print(f"→ Hay {faltan} archivos .edf que aún no tienen sus dos .fif generados.")
    else:
        print(f"→ Hay más pares .fif que archivos .edf, ¿duplicados?")

