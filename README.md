# Registration Using Recurrent Models
Combining Unet with LSTMs to register 2D slices in a 3D image.


## Installation:
------------
Start by cloning this repositiory:
```
git clone https://github.com/Armin-Saadat/master-thesis.git
cd master-thesis
```
And install the dependencies:
```
pip install ./source-code/pystrum
pip install ./source-code/neurite
pip install ./source-code/voxelmorph
```

## Train:

```
python3.7 train-scripts/<file-name>
file-name options: 
[2d.py, fc_bottleneck.py, conv_bottleneck.py, conv_all_layers.py]
```

## Evaluate:
  
```
python3.7 eval-scripts/<file-name>
file-name options:
[2d_eval.py, fc_bottleneck_eval.py, conv_bottleneck_eval.py, conv_all_layers_eval.py]
```

####
In each file, there is an argument section specified with comments at Args. Using Args, you can set the hyper-parameters of the model and determine the path for saving and loading the trained models.


