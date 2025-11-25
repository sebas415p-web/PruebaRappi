# PruebaRappi
Sistema de Análisis Inteligente para Operaciones Rappi

Descripción
Proyecto desarrollado en Python y Streamlit para analizar métricas operativas de Rappi, con:

- Bot conversacional para queries en lenguaje natural (MVP basado en reglas)
- Explorador interactivo de datos
- Insights automáticos de anomalías y tendencias
- Visualizaciones dinámicas con Plotly

Estructura del proyecto

sistema-rappi/
├── app.py                     # Script principal de la app Streamlit
├── requirements.txt           # Librerías necesarias
├── README.txt                 # Esta documentación (formato texto)
├── data/
│   └── Sistema-de-Analisis-...csv  # Dataset de métricas (no versionado en git)
├── utils.py                   # Funciones auxiliares (transformaciones, métricas)
├── config.py                  # Variables globales y configuración
├── test_analysis.py           # Tests unitarios
└── .gitignore                 # Archivos ignorados en Git

Cómo ejecutar el proyecto

Requisitos

- Python 3.10 o superior
- Conexión a internet para descargar dependencias

Pasos

1. Clona este repositorio o descarga el código a tu máquina.

2. Navega a la carpeta sistema-rappi en terminal.

3. Crea entorno virtual:

python -m venv venv

4. Activa el entorno (PowerShell):

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

O CMD:

venv\Scripts\activate.bat

5. Instala dependencias:

pip install -r requirements.txt

6. Coloca el archivo CSV en la carpeta data/.

7. Ejecuta aplicación:

streamlit run app.py

8. En la barra lateral carga el CSV y usa las pestañas para explorar y consultar datos.

Explicación de la solución

- app.py contiene la interfaz principal con 3 pestañas:
  - Explorador de datos
  - Insights automáticos
  - Bot conversacional
- Los datos CSV se transforman a formato 'long' para análisis temporal
- Funciones auxiliares en utils.py hacen el cálculo de anomalías y tendencias
- Bot conversacional responde preguntas específicas usando reglas y busca en datos
- El sistema es modular para facilitar futuras mejoras (ej., integración GPT)
- 
Referencias

- Streamlit Docs: https://docs.streamlit.io/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Plotly Python: https://plotly.com/python/
- OpenAI API (futuro para integración GPT): https://platform.openai.com/docs
