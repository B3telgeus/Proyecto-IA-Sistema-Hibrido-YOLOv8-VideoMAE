from ultralytics import YOLO

def main():
    # Load the pre-trained baseline model
    print("Loading YOLOv8n baseline model...")
    model = YOLO('yolov8n.pt')
    
    # Train the model using the provided dataset
    print("Starting training...")
    results = model.train(
        data=r'C:\Users\Santiago Bejarano\Downloads\dataset\dataset.yaml',
        epochs=50,       # You can adjust the number of epochs
        imgsz=640,
        project='results',
        name='baseline_training'
    )
    print("Training completed successfully.")

if __name__ == '__main__':
    main()
