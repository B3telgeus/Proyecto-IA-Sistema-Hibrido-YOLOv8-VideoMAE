<<<<<<< HEAD
# Proyecto de IA: Detección de Evasión en SITP / TransMilenio

## Ejecutar la Demo Unificada
Para lanzar la interfaz de usuario basada en Gradio, ejecute:

```bash
python app/app_unificada.py
```
Una vez iniciado, podrá cargar clips de video desde el navegador y observar la detección de evasión en tiempo real.

## Arquitectura Técnica
El sistema opera bajo un esquema de Inferencia por Disparo (Trigger-based inference):

- **Detección Espacial:** YOLOv8 localiza a los pasajeros y monitorea su interacción con una zona de interés (ROI) definida por una Línea de Validación.
- **Análisis Temporal:** Al detectarse un cruce, se extrae un clip de 16 frames que es procesado por un Transformer de Video (VideoMAE).
- **Clasificación de Acción:** El modelo clasifica la acción entre: Pasajero Regular, Evasión por Salto, Evasión por Puerta o Conducta Sospechosa.

## Estructura del Directorio
```text
PROYECTO-IA-Deteccion-de-Evasion-en-TRANSMILENIO/
├── app/
│   └── app_unificada.py      <-- demo de Gradio
├── models/
│   ├── best.pt               <-- YOLO entrenado
│   └── videomae_checkpoint/  <-- Carpeta de VideoMAE
├── notebooks/
│   ├── 01_Baseline.ipynb
│   └── 03_Avanzado.ipynb
├── data_samples/             <-- Los 3 videos de prueba (Normal/Salto/Error)
├── README.md
├── requirements.txt
└── .gitignore
```

## Resultados y Evaluación (Corte 3)
- **Baseline:** YOLOv8n alcanzó una precisión robusta en detección de presencia pero presentaba ambigüedad en la clasificación de la intención del movimiento.
- **Modelo Final:** La integración de VideoMAE redujo los falsos negativos en la clase "Salto" en un porcentaje significativo, permitiendo distinguir dinámicas de movimiento que son imperceptibles en un solo frame.

**Fallas Conocidas:** El sistema presenta limitaciones en escenarios de oclusión extrema (multitudes densas) y condiciones de iluminación críticas, las cuales están documentadas en el informe técnico.

## Video de Demostración
Puede ver la explicación completa de la arquitectura, el proceso de entrenamiento local y la demo funcional en el siguiente enlace:
**

## Licencia MIT
Este proyecto fue desarrollado con fines académicos para la Universidad Jorge Tadeo Lozano.
=======
# Proyecto-IA-Sistema-Hibrido-YOLOv8-VideoMAE
Sistema de vigilancia proactiva para TransMilenio basado en una arquitectura híbrida de IA. Implementa YOLOv8 para detección de sujetos y VideoMAE (Transformers) para la clasificación temporal de conductas de evasión y seguridad en estaciones.
>>>>>>> f556b0d95ec2819ad4c1bb3abb3d2b0ae9e4269e
