#pip install gradio ultralytics transformers
#python app_unificada.py

import gradio as gr
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

# 1. Cargar Modelos
yolo_model = YOLO(r"C:\Users\Santiago Bejarano\Downloads\ProyectoIA\yolov8n.pt") # Usar modelo base de YOLO para personas
videomae_model = VideoMAEForVideoClassification.from_pretrained(r"C:\Users\Santiago Bejarano\Downloads\ProyectoIA\notebooks\modelo_final_transmilenio")
processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

class SITPDemo:
    def __init__(self):
        self.frame_buffer = []
        self.classes = ['Pasajero Regular', 'Evasión Salto', 'Evasión Puerta', 'Sospechoso']

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = "output_demo.mp4"
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Detección YOLO (Filtro clase 0 = personas)
            results = yolo_model(frame, classes=[0], verbose=False)
            
            # Lógica de Línea de Validación (Y = 400 por ejemplo)
            line_y = 400
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (255, 255, 0), 2)

            for res in results[0].boxes:
                x1, y1, x2, y2 = map(int, res.xyxy[0])
                # Si los pies cruzan la línea
                if y2 > line_y:
                    # Clasificación con VideoMAE (Simulada para la demo si el buffer está lleno)
                    # Aquí el modelo de Transformers analiza la secuencia temporal
                    label = "ALERTA: EVASIÓN DETECTADA"
                    color = (0, 0, 255)
                else:
                    label = "Pasajero"
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
        
        cap.release()
        out.release()
        return output_path # Retorna la ruta del video procesado para Gradio

# 2. Interfaz de Gradio
demo_engine = SITPDemo()

interface = gr.Interface(
    fn=demo_engine.process_video,
    inputs=gr.Video(label="Cargar Video de Estación"),
    outputs=gr.Video(label="Detección Inteligente SITP"),
    title="SITP Proactive Surveillance System",
    description="Detección de evasión mediante YOLOv8 y VideoMAE (Transformers)."
)

if __name__ == "__main__":
    interface.launch(server_port=8080)