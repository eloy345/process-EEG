import os
import mne
from cargar_datos_raw import diccionario_datos_procesados_por_tipo
from mne.preprocessing import annotate_muscle_zscore
import pandas as pd

# PREPROCESADO PARA ANÁLISIS EN POTENCIALES EVOCADOS

# PREPROCESADO PARA ANÁLISIS EN BANDAS DE FRECUENCIA


def preprocesado_bandas(ruta_fif, ruta_base_crop, ruta_base_guardado):
    """
    Preprocesado simplificado para análisis de bandas de frecuencia:
    - Filtrado 1–40 Hz
    - Filtro notch 50 Hz
    - Detección de artefactos musculares
    - Proyecciones SSP (1 componente)
    - Referencia promedio
    - Guardado estructurado
    """
    print(f"🧪 Procesando: {ruta_fif}")
    raw = mne.io.read_raw_fif(ruta_fif, preload=True)

    # === 1. Filtrado banda 1–40 Hz + notch 50 Hz
    raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin')
    raw.notch_filter(freqs=50)

    # === 2. Artefactos musculares
    anotaciones, _ = annotate_muscle_zscore(raw, threshold=5.0, ch_type='eeg')
    raw.set_annotations(anotaciones)
    raw = raw.copy().annotate_bad_segments()

    # === 3. Proyecciones SSP
    projs, _ = mne.preprocessing.compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=1)
    raw.add_proj(projs)
    raw.apply_proj()

    # === 4. Referencia promedio
    raw.set_eeg_reference('average', projection=False)

    # === 5. Guardado
    ruta_relativa = os.path.relpath(ruta_fif, ruta_base_crop)
    ruta_guardado = os.path.join(ruta_base_guardado, ruta_relativa)
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
    raw.save(ruta_guardado, overwrite=True)
    print(f"✅ Guardado en: {ruta_guardado}")

def preprocesado_erps(ruta_fif, ruta_base_crop, ruta_base_guardado):
    """
    Preprocesado para análisis de ERPs:
    - Filtrado suave (0.1 - 30 Hz)
    - Interpolación de canales malos
    - Referencia promedio
    - Guardado en EEG_PROCESSED_ERPS replicando estructura de carpetas
    """
    print(f"Procesando: {ruta_fif}")
    raw = mne.io.read_raw_fif(ruta_fif, preload=True)

    # 1. Filtro pasa banda
    raw.filter(l_freq=0.1, h_freq=30.0)

    # 2. Interpolación de canales malos
    raw.interpolate_bads(reset_bads=True)

    # 3. Referencia promedio
    raw.set_eeg_reference('average', projection=False)

    # 4. Guardado respetando estructura
    ruta_relativa = os.path.relpath(ruta_fif, ruta_base_crop)
    ruta_guardado = os.path.join(ruta_base_guardado, ruta_relativa)
    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)

    raw.save(ruta_guardado, overwrite=True)
    print(f"Guardado en: {ruta_guardado}")


# PREPROCESADO PARA ANÁLISIS DE CONECTIVIDAD

def preprocesado_conectividad(ruta_fif: str, ruta_guardado_base: str, ruta_base_origen: str):
    """
    Preprocesado para análisis de conectividad EEG:
    - Detección automática de artefactos musculares.
    - Proyecciones SSP si se detectan artefactos.
    - Filtro paso banda 1-40 Hz.
    - Interpolación de canales malos.
    - Referencia promedio.
    - Guardado en EEG_PROCESSED_CONECTIVIDAD respetando estructura de carpetas.
    """
    raw = mne.io.read_raw_fif(ruta_fif, preload=True)

    # 1. Detección automática de artefactos musculares
    raw_copy = raw.copy().pick_types(eeg=True)
    anotaciones = mne.preprocessing.annotate_muscle_zscore(
        raw_copy, threshold=4.0, min_length_good=0.2
    )
    raw.set_annotations(anotaciones)

    # 2. Aplicar SSP si hay artefactos
    if len(anotaciones) > 0:
        projs, _ = mne.preprocessing.compute_proj_raw(raw, n_grad=0, n_mag=0, n_eeg=2)
        raw.add_proj(projs)
        raw.apply_proj()

    # 3. Filtro 1-40 Hz
    raw.filter(l_freq=1., h_freq=40.)

    # 4. Interpolación de canales malos
    raw.info['bads'] = list(set(raw.info['bads']))  # por si acaso
    if raw.info['bads']:
        raw.interpolate_bads()

    # 5. Referencia promedio
    raw.set_eeg_reference('average')

    # 6. Guardado con la misma estructura relativa
    ruta_relativa = os.path.relpath(ruta_fif, ruta_base_origen)
    ruta_salida = os.path.join(ruta_guardado_base, ruta_relativa)
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    raw.save(ruta_salida, overwrite=True)
    print(f"✅ Guardado: {ruta_salida}")

