# SIIM-ACR Pneumothorax Segmentation 16th place solution.

##### Requirements:
<big><pre>
OpenCV
PyTorch
numpy
scipy
matplotlib
pandas
yaml
pywt
albumentations
segmentation_models_pytorch
scikit-image
NVIDIA Apex
</pre></big>

##### Run:
<big><pre>
1: Select path to current fold as SegTrainer argument in main.py
2: python main.py
</pre></big>

#### Example of Inference:
<big><pre>
from train_utils import *
trainer = SegTrainer("path_to_fold_config")
preds = tta_predictions(trainer.model, trainer.dataloader_test, device=trainer.device).copy()
df = postproc_n_convert(preds, trainer.dataloader_test.dataset.fold_keys)
df.to_csv("output.csv", index=False)
</pre></big>