import cv2 as cv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

MAX_AGE = 60
MIN_CONF = 0.4
N_INITS = 5
WEIGHTS = 'yolov8l.pt'
VIDEO = r"C:\Users\aicpl\ShipsDatasets\VideoDataset\videos\video_24.mp4"

cap = cv.VideoCapture(video)
model = YOLO(WEIGHTS)
tracker = DeepSort(max_age=MAX_AGE, n_init=N_INITS)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    detections = model.predict(frame)[0]
    classes = detections.names
    results = []
    for xmin, ymin, xmax, ymax, p, c in detections.boxes.data.tolist():
        if p > MIN_CONF:
            results.append([[int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)], p, c])
    tracks = tracker.update_tracks(results, frame=frame)
    for track in tracks:
        if track.is_confirmed():
            i = track.track_id
            c = track.get_det_class()
            xmin, ymin, xmax, ymax = track.to_ltrb(orig=True)
            p = track.get_det_conf()
            if classes[c] == 'boat':
                frame = cv.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                if p is not None:
                    frame = cv.putText(frame, f'ID: {i}, {classes[c]}: {round(p * 100, 2)}%', (int(xmin), int(ymin + 10)), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                else:
                    frame = cv.putText(frame, f'ID: {i}, {classes[c]}: 0.0%', (int(xmin), int(ymin + 10)), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    cv.imshow('Frame', frame)
    cv.waitKey(1)
cap.release()
