# Smart Factory V2

This repository contains a Python-based computer vision pipeline for quality-control inspection workflows. The code focuses on loading ground-truth labels, running object detection, and aligning bounding boxes using ICP-style matching logic.

## What is included

The project currently includes:

- YOLO-based detection helpers
- LabelMe/JSON ground-truth loading utilities
- ICP and bounding-box transformation logic
- batch inference and comparison scripts
- a main entry point for running the pipeline

## Project structure

- `code/` – main Python source files
- `models/` – expected location for model weights (not tracked)
- `sample_dataset/` – expected location for example images and labels (not tracked)
- `qc_vision_results/` – generated output images/results (ignored)
- `qc_vision_results_failed_img/` – generated failure-case outputs (ignored)

## Requirements

Install the Python dependencies with:

```bash
pip install opencv-python numpy ultralytics scipy shapely
```

## Usage

Run the main pipeline from the code directory:

```bash
cd code
python main_v1.py
```

Other useful scripts in the repository include:

- `code/batch_inference.py`
- `code/batch_comparision.py`
- `code/run_yolo.py`

## Notes

- Model files and image/data assets are intentionally not included in this repository.
- Place your own model weights under `models/` and dataset files under `sample_dataset/` before running the pipeline.
- Generated result folders are ignored by Git so the repository stays focused on source code.
