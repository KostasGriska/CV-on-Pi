import sys, time
sys.path.append('/usr/lib/python3/dist-packages')

from collections import deque, Counter
from picamera2 import Picamera2
import cv2
import numpy as np
from gpiozero import LED

# ---------------- GPIO ----------------
led_devi = LED(17)
led_sati = LED(27)

# ---------------- COCO ----------------
CAT_ID = 17
COCO_NAMES = {16: "bird", 17: "cat"}

# ---------------- Detection params ----------------
CONF_TH = 0.35
CONF_TH_TRACK = 0.20
NMS_TH = 0.45
INFER_SIZE = (300, 300)

WINDOW = 8
HITS_N = 2
HOLD_FRAMES = 6

# ---------------- SSD preprocessing ----------------
SCALE = 1.0 / 127.5
MEAN = (127.5, 127.5, 127.5)

# ---------------- SSD Net ----------------
ssd_net = cv2.dnn.readNetFromTensorflow(
    "./ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb",
    "./ssd_mobilenet_v1_coco_11_06_2017/ssd_mobilenet_v1_coco.pbtxt"
)
ssd_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
ssd_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# ---------------- Identity Net ----------------
id_net = cv2.dnn.readNetFromONNX("cats_resnet50_single.onnx")
cv2.setNumThreads(2)
cv2.setUseOptimized(True)

CAT_IDS = ["devi", "sati"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ---------------- Camera ----------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.start()
time.sleep(1.0)

print("? Cat Detector + Identity running (q to quit)")

# ---------------- Temporal smoothing ----------------
hits = deque([0] * WINDOW, maxlen=WINDOW)
hold = 0

def center_crop_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s)//2:(h + s)//2, (w - s)//2:(w + s)//2]

# ================= MAIN LOOP =================
while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    sq = center_crop_square(frame_bgr)
    sh, sw = sq.shape[:2]

    conf_th = CONF_TH_TRACK if hold > 0 else CONF_TH

    # ---------- SSD ----------
    blob = cv2.dnn.blobFromImage(
        sq, scalefactor=SCALE, size=INFER_SIZE,
        mean=MEAN, swapRB=True, crop=False
    )
    ssd_net.setInput(blob)
    detections = ssd_net.forward()

    boxes, confs, class_ids = [], [], []

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < conf_th:
            continue
        cid = int(detections[0, 0, i, 1])
        x1, y1, x2, y2 = (detections[0, 0, i, 3:7] *
                          np.array([sw, sh, sw, sh])).astype(int)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(sw - 1, x2), min(sh - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append([x1, y1, x2, y2])
        confs.append(conf)
        class_ids.append(cid)

    idxs = cv2.dnn.NMSBoxes(
        [[x1, y1, x2-x1, y2-y1] for x1, y1, x2, y2 in boxes],
        confs, 0.0, NMS_TH
    )
    keep = idxs.flatten().tolist() if len(idxs) else []

    cat_identities = []
        # ---------- Detections ----------
    for k in keep:
        x1, y1, x2, y2 = boxes[k]
        cid = class_ids[k]
        conf = confs[k]

        label = COCO_NAMES.get(cid, "obj")
        color = (255, 0, 0)

        if cid == CAT_ID:
            crop = sq[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, (224, 224))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop = crop.astype(np.float32) / 255.0
                crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
                crop = crop.transpose(2, 0, 1)[None, ...]

                id_net.setInput(crop)
                logits = id_net.forward()

                # --- Correct softmax ---
                exp = np.exp(logits - logits.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)

                pred = int(np.argmax(probs))
                confidence = float(probs[0, pred])

                if confidence > 0.6:  # confidence gate
                    label = CAT_IDS[pred]
                    cat_identities.append(label)

                color = (0, 255, 0)

        cv2.rectangle(sq, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            sq, f"{label}: {conf:.2f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )

    # ---------- Temporal smoothing ----------
    hits.append(1 if cat_identities else 0)
    hold = HOLD_FRAMES if sum(hits) >= HITS_N else max(0, hold - 1)

    # ---------- Stable identity decision ----------
    counts = Counter(cat_identities)
    led_devi.value = counts.get("devi", 0) > 0
    led_sati.value = counts.get("sati", 0) > 0

    # ---------- UI ----------
    status = "CAT DETECTED" if hold > 0 else "Scanning..."
    cv2.rectangle(sq, (10, 10), (280, 45),
                  (0, 255, 0) if hold > 0 else (0, 0, 255), -1)
    cv2.putText(sq, status, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Cat Detector + ID", sq)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ---------------- Cleanup ----------------
picam2.stop()
cv2.destroyAllWindows()
led_devi.off()
led_sati.off()
