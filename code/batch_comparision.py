import cv2
import os
from pathlib import Path

# from icp_utils_v2 import transform_gt_bbox_homography_zahid2
# from vector_voting_icp import transform_gt_bbox_homography_zahid2
from qc_vision_pipeline import transform_gt_bbox_homography_zahid2
# from qc_vision_pipeline_v2 import transform_gt_bbox_homography_zahid2
from utils import load_detections_from_yolo, load_gt_from_labelme
from ultralytics import YOLO

import os
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

os.environ["QT_QPA_PLATFORM"] = "xcb"

# export QT_QPA_PLATFORM=xcb
def main():

    # ======================================================
    # Ground Truth
    # ======================================================

    gt_image_path = "../sample_dataset/Ground_truth/40.png"
    gt_json_path = "../sample_dataset/Ground_truth/40.json"

    # ======================================================
    # Target Folder
    # ======================================================

    target_folder = "../sample_dataset/failed_img/40"

    # ======================================================
    # Settings
    # ======================================================

    DISPLAY = False
    SAVE_RESULTS = True
    RESULTS_DIR = "../vector_voting_results/40"
    print("Saving to:", os.path.abspath(RESULTS_DIR))

    # ======================================================
    # Load GT
    # ======================================================

    img_gt = cv2.imread(gt_image_path)

    if img_gt is None:
        raise FileNotFoundError(
            f"Cannot load GT image: {gt_image_path}"
        )

    gt_boxes, hole_ids = load_gt_from_labelme(
        gt_json_path
    )

    print(f"GT Rectangles: {len(gt_boxes)}")

    # ======================================================
    # Load YOLO Model
    # ======================================================

    print("Loading YOLO model...")

    model = YOLO(
        "../models/SIDEOTRRH_MQ4_0.pt"
    )


    # ======================================================
    # Collect Target Images
    # ======================================================

    target_images = []

    target_images.extend(
        sorted(Path(target_folder).glob("*.png"))
    )

    target_images.extend(
        sorted(Path(target_folder).glob("*.jpg"))
    )

    target_images.extend(
        sorted(Path(target_folder).glob("*.jpeg"))
    )

    print(
        f"\nFound {len(target_images)} target images"
    )

    if len(target_images) == 0:
        print("No images found.")
        return

    # ======================================================
    # Process Images
    # ======================================================

    for idx, target_path in enumerate(target_images):
        output_filename = f"{target_path.stem}_icp{target_path.suffix}"
        print(
            f"\n[{idx + 1}/{len(target_images)}] "
            f"{target_path.name}"
        )

        img_target = cv2.imread(
            str(target_path)
        )

        if img_target is None:
            print(
                f"Failed to load: {target_path}"
            )
            continue

        # --------------------------------------------------
        # YOLO Detection
        # --------------------------------------------------

        detections = load_detections_from_yolo(
            model,
            str(target_path)
        )
        results = model(str(target_path), conf=0.1)

        # for r in results:
        #     print(f"len(r.boxes): {len(r.boxes)}")
        print(
            f"Detections: {len(detections)}"
        )

        # --------------------------------------------------
        # ICP Input Dictionaries
        # --------------------------------------------------

        gt_bbox_dict = {
            0: gt_boxes
        }

        hole_id_dict = {
            0: hole_ids
        }

        detection_dict = {
            0: detections
        }

        # --------------------------------------------------
        # Run ICP
        # --------------------------------------------------

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
                DISPLAY=DISPLAY,
                SAVE_RES=SAVE_RESULTS,
                outdir=RESULTS_DIR,
                output_filename=output_filename
            )
        )

        print(
            f"ICP Returned "
            f"{len(transformed_boxes[0])} boxes"
        )

        # --------------------------------------------------
        # Navigation
        # --------------------------------------------------

        if DISPLAY:

            print(
                "Press 'n' for next image "
                "or ESC to quit"
            )

            while True:

                key = cv2.waitKey(0) & 0xFF

                if key == ord('n'):
                    cv2.destroyAllWindows()
                    break

                elif key == 27:
                    cv2.destroyAllWindows()
                    return

    cv2.destroyAllWindows()

    print("\nFinished processing all images.")


if __name__ == "__main__":
    main()