import os
from typing import Dict

def diccionario_datos(ruta_base: str) -> Dict[str, str]:
    archivos_encontrados = {}
    for primer_nivel in os.listdir(ruta_base):
        ruta_primer_nivel = os.path.join(ruta_base, primer_nivel)
        if not os.path.isdir(ruta_primer_nivel):
            continue
        for colegio in os.listdir(ruta_primer_nivel):
            ruta_colegio = os.path.join(ruta_primer_nivel, colegio)
            if not os.path.isdir(ruta_colegio):
                continue
            for curso in os.listdir(ruta_colegio):
                ruta_curso = os.path.join(ruta_colegio, curso)
                if not os.path.isdir(ruta_curso):
                    continue
                for carpeta in os.listdir(ruta_curso):
                    if carpeta.lower() != "eeg":
                        continue
                    ruta_eeg = os.path.join(ruta_curso, carpeta)
                    for sujeto in os.listdir(ruta_eeg):
                        ruta_sujeto = os.path.join(ruta_eeg, sujeto)
                        if not os.path.isdir(ruta_sujeto):
                            continue
                        for session in os.listdir(ruta_sujeto):
                            ruta_session = os.path.join(ruta_sujeto, session)
                            if not os.path.isdir(ruta_session):
                                continue
                            for subcarpeta in os.listdir(ruta_session):
                                ruta_subcarpeta = os.path.join(ruta_session, subcarpeta)
                                if not os.path.isdir(ruta_subcarpeta):
                                    continue
                                for archivo in os.listdir(ruta_subcarpeta):
                                    if archivo.endswith(".edf"):
                                        ruta_absoluta = os.path.join(ruta_subcarpeta, archivo)
                                        ruta_relativa = os.path.relpath(ruta_absoluta, ruta_base)
                                        archivos_encontrados[ruta_relativa] = ruta_absoluta
    return archivos_encontrados


# # Para compobar que funciona
# if __name__ == "__main__":
#     ruta_base = "EEG_raw"  
#     archivos = diccionario_datos(ruta_base)

#     print(f"\nTotal de archivos .edf encontrados: {len(archivos)}\n")
#     for relativa, absoluta in archivos.items():
#         print(f"{relativa} → {absoluta}")

# # Obtener el diccionario de archivos .edf
# archivos = diccionario_datos(ruta_base)

# # Visualizar los primeros 10 resultados
# for i, (relativa, absoluta) in enumerate(archivos.items()):
#     print(f"{relativa} -> {absoluta}")
#     if i >= 9:
#         break

# # O mostrar cuántos archivos ha encontrado
# print(f"\nTotal de archivos .edf encontrados: {len(archivos)}")

# import csv

# with open("mapa_archivos_eeg.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Ruta relativa", "Ruta absoluta"])
#     for rel, abs_path in archivos.items():
#         writer.writerow([rel, abs_path])

# from collections import defaultdict
# import os
# import csv

# # Crear estructura de conteo
# conteo_por_colegio = defaultdict(lambda: {"PRE": 0, "POST": 0})

# # Recorrer el diccionario de rutas relativas
# for ruta_rel in archivos:  # archivos debe existir como diccionario {relativa: absoluta}
#     partes = ruta_rel.split(os.sep)
#     if "EEG" in partes:
#         tipo = partes[0].upper() if partes[0].upper() in ["PRE", "POST"] else None
#         colegio = partes[1] if tipo else partes[0]
#         if tipo:
#             conteo_por_colegio[colegio][tipo] += 1

# # Mostrar por pantalla
# print("\n📊 Conteo de archivos .edf por colegio y tipo (PRE/POST):\n")
# for colegio, datos in conteo_por_colegio.items():
#     print(f"{colegio:15} → PRE: {datos['PRE']:3}  |  POST: {datos['POST']:3}")

# # Exportar a CSV
# with open("conteo_edf_por_colegio.csv", "w", newline='', encoding="utf-8") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Colegio", "PRE", "POST"])
#     for colegio, datos in sorted(conteo_por_colegio.items()):
#         writer.writerow([colegio, datos["PRE"], datos["POST"]])

# print("\n✅ Archivo 'conteo_edf_por_colegio.csv' generado correctamente.")