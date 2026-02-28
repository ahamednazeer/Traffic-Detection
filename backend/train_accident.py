from ultralytics import YOLO

def train_model():
    # Load the model
    model = YOLO('yolo11n.pt') 

    # Train the model
    results = model.train(
        data='/Users/syed.ahamed/skillup/Traffic Detection/ITS_PROJECT_YOLOv8n.v3i.yolov11/data.yaml',
        epochs=50,
        imgsz=640,
        plots=True,
        save=True,
        project='runs/detect',
        name='accident_train',
        device='mps'
    )
    
    # Validate the model
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == '__main__':
    train_model()
