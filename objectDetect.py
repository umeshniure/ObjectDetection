# Author: Umesh Niure Sharma
# Date: October 23, 2022
# last modified date: October 24, 2022
import cv2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

print("Object detection started...")
yolo = cv.dnn.readNet("./cocoAndYoloFiles/yolov3.weights", "./cocoAndYoloFiles/yolov3.cfg")
classes = []
with open("cocoAndYoloFiles/coco.names", 'r') as f:
    classes = f.read().splitlines()

image = cv.imread("./images/working.jpg")
blob_img = cv.dnn.blobFromImage(image, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

yolo.setInput(blob_img)
output_layer_name = yolo.getUnconnectedOutLayersNames()
layer_output = yolo.forward(output_layer_name)

boxes = []
confidences = []
class_ids = []

for output in layer_output:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
            center_x = int(detection[0] * image.shape[0])
            center_y = int(detection[1] * image.shape[1])
            w = int(detection[0] * image.shape[0])
            h = int(detection[1] * image.shape[1])
            x = center_x - w//2
            y = center_y - h//2

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv.FONT_HERSHEY_DUPLEX
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x, y, w, h = boxes[i]

    label = str(classes[class_ids[i]])
    confid = str(round(confidences[i], 2))
    color = colors[i]

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label + " " + confid, (x, y + 20), font, 2, (255, 255, 255), 2)

    plt.imshow(image)
cv.imshow('image', image)
cv.waitKey(0)
print("Object detection stopped.")
