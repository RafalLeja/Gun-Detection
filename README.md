# Gun-Detection


## How to TrainModel
```
python scripts/train_model.py src/config/sliding_window.py
```

```python
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
```
## How to run webcam for rfdetr
```python
❯ uv run scripts/webcam_rfdetr.py src/config/rfdetr_detection.py notebooks/artifacts/rfdetr.ckpt --iou-threshold 0.3 --threshold 0.5
```
