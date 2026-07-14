from DetectionResult import DetectionResult
from gtholebox import GTHoleBox
import json


# ==========================================================
# LabelMe JSON Loader
# ==========================================================


def load_gt_from_labelme(json_file):
    gt_boxes = []
    hole_ids = []

    with open(json_file, "r") as f:
        data = json.load(f)

    for shape in data["shapes"]:

        if shape["shape_type"] != "rectangle":
            continue

        p1, p2 = shape["points"]

        x1 = min(p1[0], p2[0])
        y1 = min(p1[1], p2[1])
        x2 = max(p1[0], p2[0])
        y2 = max(p1[1], p2[1])

        label = int(shape["label"])

        # gt_boxes.append({
        #     "box": [x1, y1, x2, y2],
        #     "label": label
        # })
        gt_boxes.append(GTHoleBox([x1, y1, x2, y2], label))
        hole_ids.append(label)

    return gt_boxes, hole_ids


def load_detections_from_yolo(model, image_path, conf_threshold=0.20):

    detections = []

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False
    )

    result = results[0]

    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        cls = int(
            box.cls[0]
            .cpu()
            .numpy()
        )

        conf = float(
            box.conf[0]
            .cpu()
            .numpy()
        )

        # detections.append({
        #     "cam_id": 0,
        #     "box": [x1, y1, x2, y2],
        #     "label": cls,
        #     "conf": conf
        # })
        detections.append(
        DetectionResult(
            cam_id=0,
            box=box.xyxy[0].cpu().numpy().tolist(),
            label=int(box.cls[0].cpu().numpy()),
            conf=float(box.conf[0].cpu().numpy())
            )
        )

    return detections
