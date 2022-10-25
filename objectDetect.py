# Author: Umesh Niure Sharma
# Date: October 23, 2022
# last modified date: October 24, 2022
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

yolo = cv.dnn.readNet("./cocoAndYoloFiles/yolov3.weights", "./cocoAndYoloFiles/yolov3.cfg")
classes = []
with open("cocoAndYoloFiles/coco.names", 'r') as f:
    classes = f.read().splitlines()

camera = cv.VideoCapture(0)
# image = cv.imread("./images/working.jpg")
font = cv.FONT_HERSHEY_DUPLEX

print("Object detection started...")
while True:
    _, image = camera.read()
    height, width, _ = image.shape
    blob_img = cv.dnn.blobFromImage(image, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    yolo.setInput(blob_img)
    output_layers_names = yolo.getUnconnectedOutLayersNames()
    layer_outputs = yolo.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])
            confid = str(round(confidences[i], 2))
            color = colors[i]

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label + " " + confid, (x, y + 20), font, 1, color, 1)

            plt.imshow(image)

    cv.imshow('image', image)
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
print("Object detection finished!")
