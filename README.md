# Proyecto EEG – Análisis de señales neurofisiológicas en escolares

Este repositorio contiene el código y la estructura de trabajo del proyecto de tesis doctoral:  
**"Impacto de la actividad física sobre el rendimiento cognitivo en alumnado con altas capacidades"**.

El proyecto incluye:
- Procesamiento de señales EEG pre y post intervención de actividad física.
- Análisis de atención mediante tareas cognitivas con Vigilance Buddy.
- Comparaciones entre grupos (AACC vs no AACC) y condiciones (Pre/Post).
- Extracción de métricas EEG y medidas atencionales.
- Organización automatizada de datos y generación de visualizaciones.

---

## Estructura del repositorio

```text
proyecto_eeg/
├── EEG_raw/                   ← Datos brutos edf (NO incluidos en GitHub)
│   ├── Pre/
│   └── Post/
├── EEG_crop/                   ← Datos cortados .fif (NO incluidos en GitHub)
│   ├── Pre/
│   └── Post/ 
├── EEG_processed_bandas/                   ← Datos preprocesados .fif (NO incluidos en GitHub)
│   ├── Pre/
│   └── Post/ 
├── df_EEG/             ← DataFrames con metadatos y features de EEG
│   └── X_gamma_.csv
│   └── X_full_.csv
│   └── ...
├── results/               ← Resultados
├── scripts/               ← Scripts de carga, filtrado, análisis, ML...
│   └── cargar_datos.py
│   └── preprocesado-eeg.py
├── utils/                 ← Funciones auxiliares 
│   └── helpers.py
├── notebooks/             ← Notebooks de prueba o visualización
│   └── prueba.ipynb
├── main.py                ← Script principal del flujo completo
├── requirements.txt       ← Librerías necesarias
├── .gitignore             ← Archivos/carpetas excluidas del repo
└── README.md              ← Este archivo

```
---

## Tecnologías utilizadas

- Python 3.13+
- [MNE-Python](https://mne.tools) – procesamiento EEG
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn, Plotly
- JupyterLab

---


📬 Contacto
Eloy García-Pérez
Doctorando – Universidad de Castilla-La Mancha
Email: eloy.garciaperez@uclm.es
