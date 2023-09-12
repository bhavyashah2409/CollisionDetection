import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8l.pt')

video_path = "test3.mp4"
cap = cv.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True, conf=0.3, iou=0.9)[0].cpu()
    classes = results.names
    bboxes = results.boxes.xyxy.cpu().numpy().tolist()
    cls = results.boxes.cls.cpu().numpy().tolist()
    conf = results.boxes.conf.cpu().numpy().tolist()
    ids = results.boxes.id
    if ids is not None:
        ids = ids.cpu().numpy().tolist()
    else:
        ids = [0 for _ in bboxes]
    for index, (i, (xmin, ymin, xmax, ymax), p, c) in enumerate(zip(ids, bboxes, conf, cls)):
        frame = cv.putText(frame, f'ID: {i}, {classes[c]}: {round(p * 100, 2)}%', (int(xmin), int(ymin + 10)), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
        frame = cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        if index % 10 == 0:
            frame = cv.putText(frame, 'ALERT!', (200, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    
    cv.imshow("YOLOv8 Tracking", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
