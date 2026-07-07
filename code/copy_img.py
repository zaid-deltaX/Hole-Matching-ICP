from pathlib import Path
import shutil

source_dirs = [
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/3",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/6",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/7",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/8",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/9",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/10",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/13",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/14",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/15",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/17",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/18",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/19",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/20",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/21",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/22",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/26",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/27",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/28",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/30",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/31",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/32",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/34",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/36",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/37",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/40",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/41",
        r"/home/mohammad/projects/smart_factory/batch_results/failed_images/42",
]

target_dirs = [
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/3",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/6",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/7",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/8",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/9",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/10",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/13",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/14",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/15",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/17",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/18",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/19",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/20",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/21",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/22",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/26",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/27",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/28",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/29",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/30",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/31",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/32",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/34",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/36",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/37",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/40",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/41",
    r"/home/mohammad/projects/smart_factory/sample_dataset/target/42",
]

base_output_dir = Path("../sample_dataset/real_failed_img")

for source_dir, target_dir in zip(source_dirs, target_dirs):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Folder name: 3, 6, 7
    class_name = source_dir.name

    # Create output subfolder
    output_dir = base_output_dir / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect filenames from this source folder only
    source_files = {
        f.name
        for f in source_dir.iterdir()
        if f.is_file()
    }

    # Copy matching files from corresponding target folder
    for f in target_dir.iterdir():
        if f.is_file() and f.name in source_files:
            shutil.copy2(f, output_dir / f.name)
            print(f"Copied: {f.name} -> {output_dir}")

print("Done.")


# from pathlib import Path
# import shutil

# source_root = Path(
#     "/home/mohammad/projects/smart_factory/batch_results/failed_images"
# )

# target_root = Path(
#     "/home/mohammad/projects/smart_factory/sample_dataset/target"
# )

# output_root = Path(
#     "/home/mohammad/projects/smart_factory/sample_dataset/real_failed_img"
# )

# for source_dir in source_root.iterdir():

#     if not source_dir.is_dir():
#         continue

#     class_name = source_dir.name
#     target_dir = target_root / class_name

#     if not target_dir.exists():
#         print(f"Target folder missing: {class_name}")
#         continue

#     output_dir = output_root / class_name
#     output_dir.mkdir(parents=True, exist_ok=True)

#     copied = 0

#     for source_file in source_dir.iterdir():

#         if not source_file.is_file():
#             continue

#         target_file = target_dir / source_file.name

#         if target_file.exists():
#             shutil.copy2(target_file, output_dir / target_file.name)
#             copied += 1

#     print(f"{class_name}: copied {copied} files")

# print("Done.")