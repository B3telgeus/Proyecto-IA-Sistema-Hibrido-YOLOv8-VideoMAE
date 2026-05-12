import os
import sys
import subprocess

def install_dependencies():
    """Descarga e instala las librerías necesarias automáticamente."""
    print("Instalando librerías requeridas (transformers, datasets, pytorchvideo, etc.)...")
    packages = [
        "transformers", 
        "datasets", 
        "torchvision", 
        "evaluate", 
        "decord", 
        "accelerate"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    print("Librerías instaladas correctamente.\n")

# Instalamos dependencias al inicio del script
install_dependencies()

import torch
import numpy as np
from transformers import (
    VideoMAEForVideoClassification, 
    VideoMAEConfig, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from datasets import load_dataset
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, Resize

# ---------------------------------------------------------
# 1. Mapeo de Etiquetas (Proxy Dataset y Real)
# ---------------------------------------------------------
id2label = {
    0: "evasion_salto",      # Mapeado de 'high jump' / 'hurdling' / 'climbing'
    1: "pasajero_regular",   # Mapeado de 'walking' / 'passing through gate'
    2: "conducta_sospechosa" # Mapeado de 'running' / 'shoving'
}
label2id = {v: k for k, v in id2label.items()}

import glob
import decord
from torch.utils.data import Dataset

class TransMilenioVideoDataset(Dataset):
    """Cargador de videos locales usando decord."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_paths = []
        self.labels = []
        
        # Buscar subcarpetas (clases)
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if subdirs:
            for label_name in subdirs:
                if label_name in label2id:
                    label_idx = label2id[label_name]
                    class_dir = os.path.join(data_dir, label_name)
                    videos = glob.glob(os.path.join(class_dir, "*.mp4"))
                    self.video_paths.extend(videos)
                    self.labels.extend([label_idx] * len(videos))
        else:
            # Si no hay carpetas de clases, carga todo y asigna clase 0 (evasion_salto) por defecto
            print(f"ADVERTENCIA: No se encontraron subcarpetas en {data_dir}. Asignando clase 0 (evasion_salto) a todos los videos.")
            videos = glob.glob(os.path.join(data_dir, "*.mp4"))
            self.video_paths.extend(videos)
            self.labels.extend([0] * len(videos))
            
    def __len__(self):
        return len(self.video_paths)
        
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Leer video con decord
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        num_frames = len(vr)
        # Extraer 16 frames distribuidos uniformemente
        frame_indices = np.linspace(0, num_frames - 1, 16, dtype=int)
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # VideoMAE espera el tensor en formato (num_frames, num_channels, height, width)
        # El numpy de decord sale en (T, H, W, C)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        
        if self.transform:
            frames = self.transform(frames)
            
        # El Trainer de HF VideoMAE espera 'pixel_values' y 'labels'
        return {"pixel_values": frames, "labels": label}


# ---------------------------------------------------------
# 2. Configuración de Aumentación de Datos (VideoMAE)
# ---------------------------------------------------------
# Aumentación para "estirar" los pocos videos reales
# El tensor que entra es (T, C, H, W)
train_transform = Compose([
    Lambda(lambda x: x / 255.0),  # Normalización [0, 1]
    RandomHorizontalFlip(p=0.5),  # Duplica tus videos con efecto espejo
    Resize((224, 224), antialias=True), # Redimensionar al tamaño del modelo
])

# Para validación no hacemos Flip, solo recortamos y normalizamos
val_transform = Compose([
    Lambda(lambda x: x / 255.0),
    Resize((224, 224), antialias=True),
])

def apply_train_transforms(examples):
    # Función dummy para aplicar las transformaciones al dataset de HF
    examples["video"] = [train_transform(v) for v in examples["video"]]
    return examples

# ---------------------------------------------------------
# 3. Funciones para Manejar el Modelo y Transfer Learning
# ---------------------------------------------------------
def prepare_model_for_fine_tuning(model, strategy="frozen_backbone"):
    """Congela las capas base para que solo el cabezal aprenda inicialmente."""
    if strategy == "frozen_backbone":
        # Congelamos el encoder (ViT) y solo entrenamos el cabezal de clasificación
        for name, param in model.videomae.named_parameters():
            param.requires_grad = False
        print("Backbone congelado. Entrenando solo el clasificador.")
    elif strategy == "unfrozen":
        for name, param in model.videomae.named_parameters():
            param.requires_grad = True
        print("Backbone descongelado completamente.")
    return model

# Callback personalizado para descongelar el modelo en la época 5
class UnfreezeCallback(TrainerCallback):
    def __init__(self, unfreeze_epoch=5):
        self.unfreeze_epoch = unfreeze_epoch

    def on_epoch_begin(self, args, state, control, model, **kwargs):
        if state.epoch == self.unfreeze_epoch:
            print(f"\n--- Época {state.epoch}: Descongelando el backbone progresivamente ---")
            prepare_model_for_fine_tuning(model, strategy="unfrozen")
            # En la práctica, idealmente se ajusta también el learning rate aquí,
            # pero el Trainer lo maneja mediante el scheduler (warmup y decaimiento).

# ---------------------------------------------------------
# 4. Pipeline Principal (Las Dos Fases)
# ---------------------------------------------------------
def main():
    print("=== INICIANDO PIPELINE DE TRANSFER LEARNING ===")

    # Cargar modelo base pre-entrenado
    model_ckpt = "MCG-NJU/videomae-base" # Modelo base de VideoMAE
    print(f"Cargando configuración base: {model_ckpt}")
    
    config = VideoMAEConfig.from_pretrained(
        model_ckpt,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        config=config,
        ignore_mismatched_sizes=True # Permite adaptar el cabezal a 3 clases
    )

    # ---------------------------------------------------------
    # FASE 1: Calentamiento con Dataset Proxy (Kinetics / UCF)
    # ---------------------------------------------------------
    print("\n--- FASE 1: Calentamiento con Dataset Proxy ---")
    model = prepare_model_for_fine_tuning(model, strategy="frozen_backbone")

    # Aquí cargarías el subconjunto de Kinetics/UCF101 usando load_dataset
    # proxy_dataset = load_dataset("kinetics400", split="train") 
    # dataset_proxy_train = proxy_dataset.filter(lambda x: x['label'] in [proxy_classes])
    
    # IMPORTANTE: Estos datasets están comentados porque dependen de la ruta local
    # dataset_proxy_train = ... 
    # dataset_proxy_eval = ...

    args_phase1 = TrainingArguments(
        output_dir="./checkpoints_fase1_proxy",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,  # LR estándar para ajustar solo el cabezal
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,  # 5 épocas para ajustar el cabezal a los movimientos proxy
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    print("Configurando Trainer para Fase 1...")
    # trainer_phase1 = Trainer(
    #     model=model,
    #     args=args_phase1,
    #     train_dataset=dataset_proxy_train,
    #     eval_dataset=dataset_proxy_eval,
    # )
    # trainer_phase1.train()
    
    # Simulamos el guardado de la Fase 1
    # trainer_phase1.save_model("./checkpoints_fase1_proxy/best_model")
    print("Fase 1 completada (Código de entrenamiento comentado en el script para evitar errores por falta de dataset).")

    # ---------------------------------------------------------
    # FASE 2: Fine-Tuning con Videos Reales de TransMilenio
    # ---------------------------------------------------------
    print("\n--- FASE 2: Fine-Tuning Final (Dataset TransMilenio) ---")
    
    # Se simula cargar el modelo que se guardó en la Fase 1
    # model = VideoMAEForVideoClassification.from_pretrained("./checkpoints_fase1_proxy/best_model")
    
    # En las primeras 5 épocas mantenemos congelado (ya viene así de la Fase 1)
    # En la época 5 descongelamos usando el callback.
    args_phase2 = TrainingArguments(
        output_dir="./checkpoints_fase2_transmilenio",
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,  # LR MUCHO MÁS BAJO para no olvidar la Fase 1
        per_device_train_batch_size=2, # Batch más pequeño por si los videos son grandes
        num_train_epochs=10, # 10 épocas en total
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
    )

    # Cargar los videos locales desde la nueva ruta con subcarpetas
    real_data_path = r"C:\Users\Santiago Bejarano\Downloads\dataset_video"
    dataset_real_train = TransMilenioVideoDataset(data_dir=real_data_path, transform=train_transform)
    # Si tuvieras una carpeta "val", la asignarías aquí. Por ahora usamos la misma para evitar errores.
    dataset_real_eval = TransMilenioVideoDataset(data_dir=real_data_path, transform=val_transform)

    print("Configurando Trainer para Fase 2 con Unfreezing Progresivo...")
    trainer_phase2 = Trainer(
        model=model,
        args=args_phase2,
        train_dataset=dataset_real_train,
        eval_dataset=dataset_real_eval,
        callbacks=[UnfreezeCallback(unfreeze_epoch=5)] # Aplica descongelamiento en época 5
    )

    trainer_phase2.train()
    trainer_phase2.save_model("./modelo_final_transmilenio")
    print("Fase 2 completada.")
    print("\n=== ENTRENAMIENTO FINALIZADO ===")

if __name__ == "__main__":
    main()
