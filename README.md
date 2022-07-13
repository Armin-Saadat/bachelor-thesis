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

Available Models:
- **2d.py**: classic 2d U-Net
- **fc_bottleneck.py**: 2d U-Net with Fully-Connected LSTM in the lowest layer
- **conv_bottleneck.py**: 2d U-Net with Convolutional LSTM in the lowest layer
- **conv_all_layers.py**: 2d U-Net with Convolutional LSTMs in all layers
```
python3.7 train-scripts/<file-name>
```

## Evaluate:
  
Available Models:
- **2d_eval.py**: classic 2d U-Net
- **fc_bottleneck_eval.py**: 2d U-Net with Fully-Connected LSTM in the lowest layer
- **conv_bottleneck_eval.py**: 2d U-Net with Convolutional LSTM in the lowest layer
- **conv_all_layers_eval.py**: 2d U-Net with Convolutional LSTMs in all layers
```
python3.7 eval-scripts/<file-name>
```

#### In each file, there is an argument section named Args. Using Args, you can set the hyper-parameters of the model and determine the path for saving and loading the trained models.


