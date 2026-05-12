# 🚇 Sistema Inteligente de Vigilancia Proactiva - SITP/TransMilenio

**Universidad Jorge Tadeo Lozano**
**Curso:** Inteligencia Artificial (2026-1S) - Corte 2
**Autores:** Santiago Bejarano Morera & Yerson Andres Perez Cadena

## 📌 Descripción del Proyecto
Este repositorio contiene el prototipo técnico para la detección automatizada en tiempo real de evasión de tarifas y conductas sospechosas en el sistema de transporte público de Bogotá. El proyecto implementa un pipeline híbrido de visión artificial combinando detección de objetos (YOLOv8) y análisis espaciotemporal (VideoMAE).

## 📁 Estructura del Repositorio

El proyecto está organizado siguiendo las mejores prácticas para experimentación en Machine Learning:

* **`/docs/`**: Contiene el `Documento_Tecnico_Corte_2.pdf` con la justificación arquitectónica, diseño del pipeline y cronograma.
* **`/notebooks/`**: Cuadernos de experimentación ejecutables en Google Colab.
    * `01_Baseline_YOLOv8.ipynb`: Implementación rápida para detección de pasajeros y buses (Baseline).
    * `02_Experimento_Tabular_PyCaret.ipynb`: Análisis de lógica espacial mediante la definición de Regiones de Interés (ROI) para detectar intrusiones.
    * `03_Avanzado_VideoSwin_HF.ipynb`: Integración de Transformers (VideoMAE) para la clasificación avanzada de acciones (delitos/agresiones).
* **`/app/`** (Próximos pasos): Interfaz de demostración construida con Gradio.

## 🚀 Instrucciones de Reproducibilidad (Google Colab)

Para evaluar los experimentos, se recomienda ejecutar los cuadernos en el entorno de **Google Colab** para evitar conflictos de dependencias locales.

### Paso 1: Cuaderno 01 (Baseline)
1.  Abre el archivo `01_Baseline_YOLOv8.ipynb` en Colab.
2.  Ejecuta la primera celda para instalar las dependencias (`ultralytics`).
3.  Sube una imagen o video corto de prueba (ej. de TransMilenio) a la carpeta base del entorno.
4.  Ejecuta el resto de las celdas para observar la detección multiclase en acción.

### Paso 2: Cuaderno 02 (Lógica ROI)
1.  Abre el archivo `02_Experimento_Tabular_PyCaret.ipynb` en Colab.
2.  Asegúrate de subir un archivo de video (ej. `video_prueba.mp4`).
3.  Al ejecutar el cuaderno, el script procesará el video, dibujará una "Línea de Validación" amarilla y cambiará el estado de la detección a rojo ("ALERTA") si una persona la cruza de forma irregular.
4.  El resultado se guardará como un nuevo archivo `.mp4` en el entorno, el cual puedes descargar.

### Paso 3: Cuaderno 03 (Clasificador de Acciones)
**⚠️ IMPORTANTE:** Este cuaderno requiere un manejo cuidadoso de las dependencias.
1.  Abre el archivo `03_Avanzado_VideoSwin_HF.ipynb` en Colab.
2.  Ejecuta **únicamente la primera celda** (Instalación de `transformers` y utilidades).
3.  **REINICIA EL ENTORNO DE EJECUCIÓN** (`Entorno de ejecución` -> `Reiniciar sesión`). Esto es fundamental para limpiar la caché de Colab.
4.  Una vez reiniciado, ejecuta la segunda celda para cargar el modelo `VideoMAEForVideoClassification`. Si se muestra el mensaje de "éxito", la configuración es correcta.

## 🛠️ Stack Tecnológico
* **Frameworks:** PyTorch, TensorFlow/Keras.
* **Modelos:** Ultralytics (YOLOv8 Nano), Hugging Face (VideoMAE).
* **Visualización:** OpenCV, Gradio.
* **Control de Versiones y Documentación:** Mermaid (Diagramas de Flujo), Markdown.

## 📊 Métricas de Éxito Actuales
* **Precisión del Baseline:** Detección estable en escenarios de alta densidad.
* **Viabilidad (FPS):** El modelo YOLOv8n demuestra capacidad de inferencia superior a 15 FPS en video, validando su potencial para operación en tiempo real.

---
*Documento generado para la evaluación del Corte 2. Abril 2026.*