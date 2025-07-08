import os

def contar_edf_en_subcarpeta(ruta_subcarpeta):
    total = 0
    for root, _, files in os.walk(ruta_subcarpeta):
        total += sum(1 for file in files if file.lower().endswith('.edf'))
    return total

# Rutas base
base_raw = "EEG_raw"
ruta_pre = os.path.join(base_raw, "PRE")
ruta_post = os.path.join(base_raw, "POST")

# Contar
total_pre = contar_edf_en_subcarpeta(ruta_pre)
total_post = contar_edf_en_subcarpeta(ruta_post)

# Mostrar resultados
print(f"📂 Archivos .edf encontrados:")
print(f"  ▶ PRE:  {total_pre}")
print(f"  ▶ POST: {total_post}")
print(f"📊 Diferencia (POST - PRE): {total_post - total_pre}")
