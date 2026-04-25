# Datasets

1. Gun Detection
[https://www.kaggle.com/datasets/ugorjiir/gun-detection]

2. weapon detection Computer Vision Model
[https://universe.roboflow.com/weapon-detection-qktol/weapon-detection-ipl7p/dataset/7]

3. Pistols Dataset
[https://public.roboflow.com/object-detection/pistols]

4. The Vibe Code
[https://www.perplexity.ai/search/znajdz-mi-fajny-dataset-do-obj-WIheDTjaTXKuhXlsYrrHuw#3]

## Local Gunmen Subset

- Path used by the dataset loader: `sources/Gunmen Dataset/All`
- Expected annotation format: YOLO txt rows in `class x_center y_center width height`
- Default class remap used by `GunmenYoloDataset`: `15 -> 0 (Human)`, `16 -> 1 (Gun)`
