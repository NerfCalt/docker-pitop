import cv2
import numpy as np

# Carica il modello di rilevamento degli oggetti pre-addestrato
modello = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Elenco delle etichette delle classi
class_labels = []
with open('coco.names', 'r') as f:
    class_labels = [line.strip() for line in f.readlines()]

# Inizializza la webcam
webcam = cv2.VideoCapture(0)

while True:
    # Leggi il frame dalla webcam
    _, frame = webcam.read()

    # Effettua il rilevamento degli oggetti
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (608,608), swapRB=True, crop=False)
    modello.setInput(blob)
    detections = modello.forward(modello.getUnconnectedOutLayersNames())

    # Elabora le rilevazioni
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                label = class_labels[class_id]
                box = obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                x, y, w, h = box.astype(int)

                # Disegna il rettangolo e l'etichetta dell'oggetto rilevato sul frame
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostra il frame con le rilevazioni
    cv2.imshow('Rilevamento oggetti', frame)

    # Interrompi l'esecuzione quando viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
webcam.release()
cv2.destroyAllWindows()
