import cv2
import numpy as np
import pyttsx3

engine = pyttsx3.init()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

cap = cv2.VideoCapture(0)

known_width = 85.60 
focal_length = 500  

def calculate_distance(object_width_in_frame):
    """Calculate distance to the object based on its width in the frame."""
    if object_width_in_frame > 0:
        distance = (known_width * focal_length) / object_width_in_frame
        return distance
    return None

objects_with_states = ["door", "window"]

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

def detect_objects(frame):
    """Perform object detection and return detected objects and their states."""
    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    object_states = {}

    if len(indexes) > 0:
        for i in indexes.flatten():
            label = str(classes[class_ids[i]])
            detected_objects.append(label)

            distance = calculate_distance(boxes[i][2])  
            if distance is not None:
                speak(f"{label} is detected at a distance of {distance:.2f} centimeters.")
                print(f"{label} is detected at a distance of {distance:.2f} cm.")

            if label in objects_with_states:
                _, _, w, _ = boxes[i]
                if w > width / 2:  
                    object_states[label] = "closed"
                else:
                    object_states[label] = "open"

            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return detected_objects, object_states, frame

while True:
    ret, frame = cap.read()

    if not ret:
        break

    detected_objects, object_states, annotated_frame = detect_objects(frame)

    for obj in detected_objects:
        if obj in object_states:
            state = object_states[obj]
            speak(f"{obj} is {state}")
        else:
            speak(f"{obj} detected")

    cv2.imshow("Webcam", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
