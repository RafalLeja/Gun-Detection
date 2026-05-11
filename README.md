# Gun-Detection


## How to TrainModel
```
python scripts/train_model.py src/config/sliding_window.py
```

```python
weights = ResNet18_Weights.DEFAULT
preprocess = weights.transforms()
```

```
python scripts/infer_sliding_window.py src/config/sliding_window.py "logs/sliding_window_baseline_3classes_20260511_213623/epoch=7-step=1184.ckpt"
```