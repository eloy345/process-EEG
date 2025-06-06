# Proyecto EEG – Análisis de señales neurofisiológicas en escolares

Este repositorio contiene el código y la estructura de trabajo del proyecto de tesis doctoral:  
**"Impacto de la actividad física sobre el rendimiento cognitivo en alumnado de educación primaria, con especial atención a estudiantes con altas capacidades"**.

El proyecto incluye:
- Procesamiento de señales EEG pre y post intervención.
- Análisis de atención mediante tareas cognitivas con Vigilance Buddy.
- Comparaciones entre grupos (AACC vs no AACC) y condiciones (Pre/Post).
- Extracción de métricas EEG y medidas atencionales.
- Organización automatizada de datos y generación de visualizaciones.

---

## Estructura del repositorio

proyecto_eeg/
├── EEG/ # Datos brutos (NO incluidos en GitHub)
├── results/ # Señales limpias, métricas, figuras
├── scripts/ # Scripts de carga, filtrado, análisis
├── utils/ # Funciones auxiliares reutilizables
├── notebooks/ # Notebooks de prueba o visualización
├── df_master/ # DataFrame con metadatos y rutas de EEG
├── main.py # Script principal del flujo completo
├── requirements.txt # Librerías necesarias
├── .gitignore # Archivos/carpetas excluidas del repo
└── README.md # Este archivo

yaml
Copiar
Editar

---

## Tecnologías utilizadas

- Python 3.11+
- [MNE-Python](https://mne.tools) – procesamiento EEG
- Pandas, NumPy, SciPy
- Matplotlib, Seaborn, Plotly
- JupyterLab

---

## Cómo empezar

1. Clona este repositorio:
   ```bash
   git clone https://github.com/eloy345/process-EEG.git
   cd process-EEG
Instala los requisitos:

bash
Copiar
Editar
pip install -r requirements.txt
Ejecuta el proyecto:

bash
Copiar
Editar
python main.py
🧪 Estado actual
 Estructura básica creada

 Sistema de carpetas implementado

 Lectura de EEG desde CSV

 Filtros y preprocesado

 Extracción de métricas

 Comparación de condiciones

📬 Contacto
Eloy García-Pérez
Doctorando – Universidad de Castilla-La Mancha
Email: tu_email@uclm.es
