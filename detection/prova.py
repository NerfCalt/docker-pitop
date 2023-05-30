import cv2
import pitop from Camera, Pitop

# Carica il modello di rilevamento degli oggetti pre-addestrato
modello = cv2.dnn.readNetFromCaffe('detection/MobileNetSSD_deploy.prototxt.txt', 'detection/MobileNetSSD_deploy.caffemodel')

# Elenco delle etichette delle classi
class_labels = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Inizializza la webcam
webcam = cv2.VideoCapture(0)
#camera = Camera()
#pitop = Pitop()

while True:
    # Leggi il frame dalla webcam, _ è un valore di ritorno della funzione webcam.read che non ci interessa 
    _, frame = webcam.read()  
    #frame=camera.get_frame(frame)
    


    # Effettua il rilevamento degli oggetti
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    modello.setInput(blob)
    detections = modello.forward()

    # Elabora le rilevazioni
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            class_id = int(detections[0, 0, i, 1])
            label = class_labels[class_id]
            box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            x, y, w, h = box.astype(int)

            # Disegna il rettangolo e l'etichetta dell'oggetto rilevato sul frame
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostra il frame con le rilevazioni
    cv2.imshow('Rilevamento oggetti', frame)

    # Interrompi l'esecuzione quando viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pitop.miniscreen.display_image(frame)
# Rilascia le risorse
webcam.release()
cv2.destroyAllWindows()
