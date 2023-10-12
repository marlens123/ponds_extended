# Detection of Melt Ponds on Arctic Sea Ice from Infrared Images

Melt ponds are pools of water on Arctic sea ice that have a strong influence on the Arctic energy budget by increasing the amount of sunlight that is absorbed. Accurate quantitative analysis of melt ponds is important for improving Arctic climate model predictions.

The objective of this repository is to develop a segmentation tool that partitions thermal infrared (TIR) images into sea ice, melt pond and ocean classes. The current model is a U-net with ResNet34 backbone, pretrained on ImageNet. So far, 10 training images are available. Annotation of additional images is in progress.

---------------------------------------
### Model Architecture

<img scr="https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729" width="200">

![model_architecture|50%](https://github.com/marlens123/ponds_extended/assets/80780236/84dde17c-6ecd-4608-af7f-7be75de84729)

---------------------------------------
### Current Results

![quali](https://github.com/marlens123/ponds_extended/assets/80780236/5f67c223-b8e2-4c26-b8ab-8ae381015a77)
*As of 31.08.2023 (10 training images): left - model input, middle - earlier model results using 8 images, right - updated model results. Grey - sea ice, black - melt ponds, white - ocean. All images are from flight 9 conducted during the PS131 ATWAICE campaign [1].*


----------------------------------------

To make the code work, do the following:

1. Create a new anaconda environment with Python 3.10.0
2. pip install the packages specified in `requirements.txt` (part 1)
3. If you try to run `model_selection.py` (recreation of training pipeline), go to ADD1 and ADD2 further below in this file

---------------------------------------

The weights of the current final model can be found in `weights/`. 

`model_weights/` contains the weights of the thesis model. This folder also contains the weights of a patch size 32 x 32 configuration, which was used to test smooth patch prediction.

In the file heads you will find relative paths to data that you need for running the code. These paths refer to the myshare folder that comes with the thesis.


Files to recreate experiments:
---------------------------------
- `model_selection.py`: to recreate the model training (runs `train.py` with different configurations) 
- `train.py`: contains our training pipeline (you may need to adjust the wandb login in this file)

- `data_preparation/extract.ipynb`: to extract TIR images for inspection
- `data_preparation/edge_detection.ipynb`: to create edge maps used for annotation
- `prepare_data.py`: to create prepared np arrays from .nc files
- `qualitative_evaluation.ipynb`: to recreate our qualitative evaluation predictions
- `predict_image.py`: contains the function used for prediction
- `mpf.ipynb`: to recreate our melt pond fraction computations


Additional file contents:
-------------------------
`visualize.ipynb` was used to create visualizations in our thesis
`stats.ipynb` was used to create statistics for our thesis

`utils/` contains 
- `smooth_tiled_predictions.py`: patch stitching function that we integrated in our prediction function. We copied the content of this file from https://github.com/bnsreenu/python_for_microscopists/tree/master/229_smooth_predictions_by_blending_patches. For more information on the reference, see the file head
- `patch_extraction.py`: for patch extraction (used in `train.py`)
- `data.py`: to load data for model training (used in `train.py`), inspired by https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb 
- `augmentation.py`: contains augmentation and preprocessing function. Uses albumentation library (https://github.com/albumentations-team/albumentations)
- `image_transform.py`: contains functions for image transformation


`models/segmentation_models_qubvel/`: contains the segmentation models repository (https://github.com/qubvel/segmentation_models). We added the option to train with dropout layers in `segmentation_models/models/unet.py` (marked in file with 'CHANGED')

`vis_segmentation/` contains the OSSP classification algorithm (repo downloaded from https://github.com/wrightni/OSSP)


### <a id="Error1"></a>ADD1: If you get the error `ImportError: cannot import name 'get_submodules_from_kwargs'`

(see https://github.com/qubvel/segmentation_models/issues/248)

- navigate to your environment directory, and then to `...\Lib\site-packages\classification_models\__init__.py`
- if this is empty, insert the following code:

------------------------------------------------------
```python
import keras_applications as ka
from .__version__ import __version__

def get_submodules_from_kwargs(kwargs):
    backend = kwargs.get('backend', ka._KERAS_BACKEND)
    layers = kwargs.get('layers', ka._KERAS_LAYERS)
    models = kwargs.get('models', ka._KERAS_MODELS)
    utils = kwargs.get('utils', ka._KERAS_UTILS)
    return backend, layers, models, utils
```
------------------------------------------------------

### <a id="Error2"></a>ADD2: To be able login to wandb account

- before executing `model_selection.py`or `train.py`, execute the following command: `wand login` and insert the key that was sent via email
- you can also modify the scripts and login to your own wandb account

------------------------------------------------------

To run the OSSP classification algorithm (https://github.com/wrightni/OSSP), do the following:

1. Create a new anaconda environment with Python 3.6
2. conda install the packages specified in `requirements.txt` (part 2)
3. Follow the instruction in `./vis_segmentation/OSSP-wright/readme.md`

(installation may not work properly due to changed package dependencies. I recommend using the classified images, transferred shared in the myshare folder. This is relevant for `mpf.ipynb`)

----------------------------
References:
-----------------------------
[1] Kanzow, Thorsten (2023). The Expedition PS131 of the Research Vessel POLARSTERN to the
Fram Strait in 2022. Ed. by Horst Bornemann and Susan Amir Sawadkuhi. Bremerhaven. DOI: 10.57738/BzPM\_0770\_2023.

- wandb and k-crossfold validation: https://www.kaggle.com/code/ayuraj/efficientnet-mixup-k-fold-using-tf-and-wandb/notebook
- patchify library for patch extraction: https://pypi.org/project/patchify/
- patch prediction: https://github.com/bnsreenu/python_for_microscopists/blob/master/206_sem_segm_large_images_using_unet_with_custom_patch_inference.py


The project is the extended version of my Bachelor thesis with the Polar Remote Sensing Group, University of Bremen and the Computer Vision Group, University of Osnabrück (submission 08/2023).
