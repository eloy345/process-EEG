import os
from preprocesado_eeg import preprocesado_bandas
from cargar_datos_raw import diccionario_datos_procesados_por_tipo
import mne

#  CONFIGURACIÓN DE RUTAS - Mis archvos estan EEG_crop/PRE/COLEGIO/CURSO/EEG/SUJETO/Session01/RecordingS01R000_EDF/aacc_basal.fif

ruta_base_crop = "EEG_crop"
ruta_base_guardado = "EEG_processed_bandas"


# Cargar las rutas de señales tipo basal
archivos = diccionario_datos_procesados_por_tipo(ruta_base_crop, "basal") #ejecutar con basal y vb


print(f"📁 Buscando señales en: {os.path.abspath(ruta_base_crop)}")
print(f"📁 Guardando resultados en: {os.path.abspath(ruta_base_guardado)}")
print(f"🔍 Total señales encontradas: {len(archivos)}")

for ruta_relativa, ruta_absoluta in archivos.items():
    try:
        preprocesado_bandas(ruta_absoluta, ruta_base_crop, ruta_base_guardado)
    except Exception as e:
        print(f"❌ Error procesando {ruta_relativa}: {e}")
        raw = mne.io.read_raw_fif(ruta_absoluta, preload=False)
        print("📋 Nombres de canales:", raw.info['ch_names'])
        print("Canales malos:", raw.info['bads'])
        print("📋 Tipos de canales:", raw.get_channel_types())
        break