import cv2
import os
from pathlib import Path

from qc_vision_pipeline_2 import transform_gt_bbox_homography_zahid2
from utils import load_detections_from_yolo, load_gt_from_labelme
from ultralytics import YOLO

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
os.environ["QT_QPA_PLATFORM"] = "xcb"

# ======================================================
# CONFIGURATION
# ======================================================

GT_ROOT = "../sample_dataset/Ground_truth"
TARGET_ROOT = "/home/mohammad/projects/smart_factory/sample_dataset/target/"
OUTPUT_ROOT = "../qc_vision_results"

DISPLAY = False
SAVE_RESULTS = True


def collect_images(folder):
    """Collect all image files from a folder."""
    images = []

    images.extend(sorted(folder.glob("*.png")))
    images.extend(sorted(folder.glob("*.jpg")))
    images.extend(sorted(folder.glob("*.jpeg")))

    return images


def main():

    # ======================================================
    # Load YOLO Once
    # ======================================================

    print("Loading YOLO model...")

    model = YOLO(
        "../models/SIDEOTRRH_MQ4_0.pt"
    )

    # ======================================================
    # Get Target Folders (1,2,3...42)
    # ======================================================

    target_folders = sorted(
        [p for p in Path(TARGET_ROOT).iterdir() if p.is_dir()],
        key=lambda x: int(x.name)
    )

    print(
        f"Found {len(target_folders)} target folders"
    )

    # ======================================================
    # Process Each Folder
    # ======================================================

    for target_folder in target_folders:

        folder_name = target_folder.name

        print("\n" + "=" * 80)
        print(f"Processing Folder: {folder_name}")
        print("=" * 80)

        # --------------------------------------------------
        # Find Corresponding GT Image + JSON
        # --------------------------------------------------

        gt_image_path = (
            Path(GT_ROOT) / f"{folder_name}.png"
        )

        if not gt_image_path.exists():
            gt_image_path = (
                Path(GT_ROOT) / f"{folder_name}.jpg"
            )

        if not gt_image_path.exists():
            gt_image_path = (
                Path(GT_ROOT) / f"{folder_name}.jpeg"
            )

        gt_json_path = (
            Path(GT_ROOT) / f"{folder_name}.json"
        )

        if not gt_image_path.exists():
            print(
                f"GT image not found for folder "
                f"{folder_name}"
            )
            continue

        if not gt_json_path.exists():
            print(
                f"GT json not found for folder "
                f"{folder_name}"
            )
            continue

        # --------------------------------------------------
        # Output Folder
        # --------------------------------------------------

        output_folder = (
            Path(OUTPUT_ROOT) / folder_name
        )

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            # ======================================================
            # Load GT Image
            # ======================================================

            img_gt = cv2.imread(
                str(gt_image_path)
            )

            if img_gt is None:
                print(
                    f"Failed to load GT image: "
                    f"{gt_image_path}"
                )
                continue

            # ======================================================
            # Load GT Labels
            # ======================================================

            gt_boxes, hole_ids = (
                load_gt_from_labelme(
                    str(gt_json_path)
                )
            )

            print(
                f"GT Boxes: {len(gt_boxes)}"
            )

            # ======================================================
            # Collect Target Images
            # ======================================================

            target_images = collect_images(
                target_folder
            )

            print(
                f"Target Images: "
                f"{len(target_images)}"
            )

            if len(target_images) == 0:
                print(
                    f"No images found in "
                    f"{target_folder}"
                )
                continue

            # ======================================================
            # Process All Images in Folder
            # ======================================================

            for idx, target_path in enumerate(
                target_images
            ):

                print(
                    f"\n[{idx + 1}/"
                    f"{len(target_images)}] "
                    f"{target_path.name}"
                )

                # ----------------------------------------------
                # Load Target Image
                # ----------------------------------------------

                img_target = cv2.imread(
                    str(target_path)
                )

                if img_target is None:
                    print(
                        f"Failed to load: "
                        f"{target_path}"
                    )
                    continue

                # ----------------------------------------------
                # YOLO Detection
                # ----------------------------------------------

                detections = (
                    load_detections_from_yolo(
                        model,
                        str(target_path)
                    )
                )

                print(
                    f"Detections: "
                    f"{len(detections)}"
                )

                # ----------------------------------------------
                # Prepare Dictionaries
                # ----------------------------------------------

                gt_bbox_dict = {
                    0: gt_boxes
                }

                hole_id_dict = {
                    0: hole_ids
                }

                detection_dict = {
                    0: detections
                }

                output_filename = (
                    f"{target_path.stem}_icp"
                    f"{target_path.suffix}"
                )

                # ----------------------------------------------
                # Run ICP / Homography
                # ----------------------------------------------

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
                        outdir=str(output_folder),
                        output_filename=output_filename
                    )
                )

                print(
                    f"ICP Returned "
                    f"{len(transformed_boxes[0])} boxes"
                )

                # ----------------------------------------------
                # Optional Display
                # ----------------------------------------------

                if DISPLAY:

                    print(
                        "Press 'n' for next image "
                        "or ESC to quit"
                    )

                    while True:

                        key = (
                            cv2.waitKey(0)
                            & 0xFF
                        )

                        if key == ord("n"):
                            cv2.destroyAllWindows()
                            break

                        elif key == 27:
                            cv2.destroyAllWindows()
                            return

        except Exception as e:

            print(
                f"Error processing folder "
                f"{folder_name}: {e}"
            )

    cv2.destroyAllWindows()

    print(
        "\nFinished processing all folders."
    )


if __name__ == "__main__":
    main()