import cv2 as cv
from ultralytics import YOLO

video = 'test7.mp4'
cap = cv.VideoCapture(video)

model = YOLO('yolov8l.pt')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, conf=0.4, iou=0.4, persist=True)[0]
    classes = results.names
    results = results.boxes
    xyxy = results.xyxy.cpu().numpy().tolist()
    cls = results.cls.cpu().numpy().tolist()
    conf = results.conf.cpu().numpy().tolist()
    ids = results.id
    if ids is not None:
        for i, (xmin, ymin, xmax, ymax), p, c in zip(ids, xyxy, conf, cls):
            frame = cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            frame = cv.putText(frame, f'ID:{i} {classes[c]}: {round(p * 100, 2)}', (int(xmin), int(ymin + 10)), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    else:
        for (xmin, ymin, xmax, ymax), p, c in zip(xyxy, conf, cls):
            frame = cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            frame = cv.putText(frame, f'{classes[c]}: {round(p * 100, 2)}', (int(xmin), int(ymin + 10)), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv.imshow('Frame', frame)
    cv.waitKey(1)
cap.release()
