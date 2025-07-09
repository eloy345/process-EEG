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


def diccionario_datos_procesados_por_tipo(ruta_base: str, tipo: str) -> Dict[str, str]:
    """
    Recorre EEG_processed y devuelve un diccionario con rutas a archivos .fif
    que contengan 'aacc_basal.fif' o 'aacc_vb.fif' según el tipo solicitado.

    Parámetros:
    -----------
    ruta_base : str
        Ruta base del proyecto (debe contener PRE/POST/...)
    tipo : str
        'basal' o 'vb' para filtrar el tipo de archivo.

    Retorna:
    --------
    Dict[str, str]
        Diccionario con {ruta_relativa: ruta_absoluta}
    """
    assert tipo in ["basal", "vb"], "Tipo debe ser 'basal' o 'vb'"
    archivos_filtrados = {}

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

                    ruta_eeg = os.path.join(ruta_curso, carpeta)
                    for sujeto in os.listdir(ruta_eeg):
                        ruta_sujeto = os.path.join(ruta_eeg, sujeto)
                        if not os.path.isdir(ruta_sujeto):
                            continue

                        for session in os.listdir(ruta_sujeto):
                            ruta_session = os.path.join(ruta_sujeto, session)
                            if not os.path.isdir(ruta_session):
                                continue

                            for carpeta_final in os.listdir(ruta_session):
                                ruta_final = os.path.join(ruta_session, carpeta_final)
                                if not os.path.isdir(ruta_final):
                                    continue

                                for archivo in os.listdir(ruta_final):
                                    if archivo.endswith(f"aacc_{tipo}.fif"):
                                        ruta_absoluta = os.path.join(ruta_final, archivo)
                                        ruta_relativa = os.path.relpath(ruta_absoluta, ruta_base)
                                        archivos_filtrados[ruta_relativa] = ruta_absoluta
    return archivos_filtrados


