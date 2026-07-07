import numpy as np
from typing import List, Optional


class DetectionResult:
    def __init__(self, cam_id: int, box: List[float], label: int, conf: float):
        self.cam_id = cam_id
        self.box = [min(box[0], box[2]), 
                    min(box[1], box[3]), 
                    max(box[0], box[2]), 
                    max(box[1], box[3])]
        self.label = label
        self.conf = conf

        """
        crack attributes
        """
        self.image_patch: Optional[np.ndarray] = None
        self.crack_classify_conf = .0
        self.crack_classify_ok = False

    def calculate_iou(self, box: List[float]) -> tuple:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        Each box is represented as (xmin, ymin, xmax, ymax).
        """
        assert(len(self.box) == 4)
        assert(len(box) == 4)

        target_box = [min(box[0], box[2]), 
                      min(box[1], box[3]), 
                      max(box[0], box[2]), 
                      max(box[1], box[3])]

       # Get the coordinates of the intersection box
        x1 = max(self.box[0], target_box[0])  # The maximum of the left (xmin)
        y1 = max(self.box[1], target_box[1])  # The maximum of the top (ymin)
        x2 = min(self.box[2], target_box[2])  # The minimum of the right (xmax)
        y2 = min(self.box[3], target_box[3])  # The minimum of the bottom (ymax)

        # Check if there is an intersection
        if x2 < x1 or y2 < y1:
            return 0.0, 0.0  # No intersection, return IoU as 0

        # Calculate intersection area
        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate the area of both boxes
        self_area = (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])
        box_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

        # Calculate the union area
        union_area = self_area + box_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        return iou, intersection_area/self_area if self_area > 0 else 0.0
    
    def get_cam_id(self) -> int:
        return self.cam_id

    def get_confidence(self) -> float:
        return self.conf

    def get_box_center(self) -> np.ndarray:    
        assert len(self.box) == 4
        return np.array([
            (self.box[0] + self.box[2]) / 2,
            (self.box[1] + self.box[3]) / 2,
            ])

    def get_crack_classify_ok(self):
        return self.crack_classify_ok

    def set_crack_classifiy_ok(self, conf: float):
        self.crack_classify_ok = True
        self.crack_classify_conf = conf
        print(f"WW) crack classify cam({self.cam_id}, box({self.box}), conf({conf})")

    def get_crack_classify_confidence(self) -> float:
        return self.crack_classify_conf
    
    def add_padding_to_patches(self):
        padding_size = 10
        
        x1 = 0 if self.box[0] - padding_size < 0 else self.box[0] - padding_size
        y1 = 0 if self.box[1] - padding_size < 0 else self.box[1] - padding_size
        
        x2 = 2473 if self.box[2] + padding_size > 2473 else self.box[2] + padding_size
        y2 = 2063 if self.box[3] + padding_size > 2063 else self.box[3] + padding_size
        
        return x1, y1, x2, y2

    def set_crack_image_patch(self, images: List[np.ndarray], add_padding: bool=False):
        [x1, y1, x2, y2] = self.box
        if int(y2) - int(y1) <= 0 or int(x2) - int(x1) <= 0:
            return None
        
        if add_padding:
            x1, y1, x2, y2 = self.add_padding_to_patches()
        
        self.image_patch = images[self.cam_id][int(y1): int(y2), int(x1):int(x2)]
        return self.image_patch
    
    def get_crack_image_patch(self):
        return self.image_patch
