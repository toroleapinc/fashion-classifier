# Fashion-MNIST Classifier

CNN and ResNet for Fashion-MNIST classification. 94.7% test accuracy with a small ResNet + squeeze-and-excitation blocks.

The main training code is in `train.py`. There's also a notebook (`exploration.ipynb`) where I tried different architectures.

## How to run

```
pip install -r requirements.txt
python train.py
python predict.py --image path/to/img.png
```

Fashion-MNIST downloads automatically via torchvision.
