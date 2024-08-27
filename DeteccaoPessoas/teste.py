import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")

# Inicializa a captura de vídeo
cap = cv2.VideoCapture('videos/classroom.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza a detecção usando o modelo YOLO
    results = model(frame)

    # Itera sobre os resultados da detecção
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do bounding box
            confidence = box.conf[0]  # Confiança da detecção
            class_id = int(box.cls[0])  # ID da classe detectada
            label = model.names[class_id]  # Nome da classe (ex: "person", "dog", etc.)

            # Filtra para detectar apenas pessoas, crianças e animais
            if label in ["person", "dog", "cat"]:
                # Define a cor do bounding box dependendo da classe
                if label == "person":
                    color = (0, 255, 0)  # Verde para pessoas
                else:
                    color = (0, 0, 255)  # Vermelho para animais

                # Desenha o bounding box e o rótulo
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibe o vídeo com as detecções
    cv2.imshow('YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
