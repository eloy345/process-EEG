import mne

ruta = "C:/Users/Eloy/OneDrive - Universidad de Castilla-La Mancha (1)/Tesis_EEG/proyecto_eeg/EEG_crop/POST/CEIP_Cristobal_Colon/5/EEG\/DMD/Session01/RecordingS01R000_EDF/aacc_vb.fif"

raw = mne.io.read_raw_fif(ruta, preload=False)

# 1. Ver nombres de canales
print("Canales:", raw.ch_names)

# 2. Ver si tiene montage asignado
print("Montage actual:", raw.get_montage())

# 3. Verificar posiciones
dig = raw.info['dig']
print("Tiene posiciones 3D:", dig is not None)
raw.plot_sensors(kind='3d', show_names=True, block=True)

import matplotlib.pyplot as plt
plt.show()