import time

import cv2

# from icp_utils_v2 import opencv_plot, opencv_plot2, transform_gt_bbox_homography_zahid2
# from vector_voting_icp import transform_gt_bbox_homography_zahid2
# from icp_utils import transform_gt_bbox_homography_zahid2
from qc_vision_pipeline import transform_gt_bbox_homography_zahid2
# from qc_vision_pipeline_2 import transform_gt_bbox_homography_zahid2

# from qt_viewer import show_results
from utils import load_detections_from_yolo, load_gt_from_labelme

from ultralytics import YOLO

import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["XDG_SESSION_TYPE"] = "wayland"
# os.environ["QT_QPA_PLATFORM"] = "wayland"
# os.environ["DISPLAY"] = ":0"     
# os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

output_path = "../results/new_gt/"
# output_path = "../results/old_icp/"
# output_path = "../results/new_icp/"

def main():

    # ------------------------------------------------------
    # Dataset Paths
    # ------------------------------------------------------
    # gt_image_path = "../sample_dataset/Ground_truth/5.png"
    # gt_json_path = "../sample_dataset/Ground_truth/5.json"

    # target_image_path = "../sample_dataset/orginal_img/5/C0004_D20250205T171040.png"
        # ------------------------------------------------------
    # gt_image_path = "../sample_dataset/Ground_truth/40.png"
    # gt_json_path = "../sample_dataset/Ground_truth/40.json"
    # target_image_path = "../sample_dataset/orginal_img/40/C0039_D20250205T171040.png"

    gt_image_path = "../sample_dataset/Ground_truth_v2/17.png"
    gt_json_path = "../sample_dataset/Ground_truth_v2/17.json"
    target_image_path = "/home/mohammad/projects/smart_factory/sample_dataset/target/17/C0020_D20250205T171040.png"
    # # ------------------------------------------------------

    img_gt = cv2.imread(gt_image_path)
    img_target = cv2.imread(target_image_path)

    if img_gt is None:
        raise FileNotFoundError(
            f"Cannot load image: {gt_image_path}"
        )

    if img_target is None:
        raise FileNotFoundError(
            f"Cannot load image: {target_image_path}"
        )

    # print("Loading GT labels...")

    gt_boxes, hole_ids = load_gt_from_labelme(
        gt_json_path
    )

    model = YOLO("../models/SIDEOTRRH_MQ4_0.pt")

    detections = load_detections_from_yolo(
        model,
        target_image_path
    )

    print(f"GT Rectangles        : {len(gt_boxes)}")
    print(f"Detection Rectangles : {len(detections)}")
    # ------------------------------------------------------
    # ICP Input Dictionaries
    # ------------------------------------------------------

    gt_bbox_dict = {
        0: gt_boxes
    }

    hole_id_dict = {
        0: hole_ids
    }

    detection_dict = {
        0: detections
    }

    # ------------------------------------------------------
    # Run ICP
    # ------------------------------------------------------

    print("Running ICP...")

    # transformed_boxes, transformed_ids = (
    #     transform_gt_bbox_homography_zahid2(
    #         total_cams=2,
    #         gt_bbox_dict=gt_bbox_dict,
    #         detection_dict=detection_dict,
    #         hole_id_dict=hole_id_dict,
    #         img_dict={
    #             0: {
    #                 "gt": img_gt,
    #                 "target": img_target
    #             }
    #         },
    #         DISPLAY=True,
    #         outdir="../results/"
    #     )
    # )
    # start = time.perf_counter()
    transformed_boxes, transformed_ids = (
        transform_gt_bbox_homography_zahid2(
            total_cams=2,
            gt_bbox_dict=gt_bbox_dict,
            detection_dict=detection_dict,
            hole_id_dict=hole_id_dict,
            img_dict={
                0: {
                    "gt": img_gt,
                    "target": img_target
                }
            },
            outdir=output_path,
            SAVE_RES=True,
            DISPLAY=True
        )
    )
    # print(f"\n[VOTING] Time: {(time.perf_counter()-start)*1000:.3f} ms")
    print(
        f"ICP Returned {len(transformed_boxes[0])} boxes"
    )


if __name__ == "__main__":
    main()