# Robustness and Overfitting Behavior of Implicit Background Models


Shirley Liu, Charles Lehman, [Ghassan AlRegib](https://www.ghassanalregib.info)


This repository contains the Jupyter Notebooks for our paper **Robustness and Overfitting Behavior of Implicit Background Models**, accepted to the IEEE International Conference on Image Processing (ICIP) in 2020.


<p align="center">
    <img src="https://github.com/sliu54/scribe-robustness-overfit/raw/master/res/subset_aug_seg.png" alt="Segmentation Masks">
</p>


## Abstract
In this paper, we examine the overfitting behavior of image classification models modified with Implicit Background Estimation (SCrIBE), which transforms them into weakly supervised segmentation models that provide spatial domain visualizations without affecting performance. Using the segmentation masks, we derive an overfit detection criterion that does not require testing labels. In addition, we assess the change in model performance, calibration, and segmentation masks after applying data augmentations as overfitting reduction measures and testing on various types of distorted images.


## Jupyter Notebooks
The following notebooks are provided:
* ***prepare_dataset.ipynb***: imports the CIFAR-10 training and testing dataset, defines the data augmentations used in the experiment, and divides the training set into balanced subsets ranging from 10% to 100% of the full training set.

* ***init_IBE.ipynb***: defines the SCrIBE model by importing the ResNet-18 architecture and making the necessary changes for it to function as a weakly supervised segmentation model. The attention function and training configurations are also defined here.

* ***init_non_IBE.ipynb***: defines the non-SCrIBE model and its training configurations.

* ***trainer.ipynb***: imports the models and training dataset, trains the dataset with varying subset sizes and data augmentations on both models and saves the trained models.

* ***tester.ipynb***: loads the saved models and evaluates the testing dataset after applying different types and levels of corruptions using [ImageNet-C](https://github.com/hendrycks/robustness). Segmentation mask and attention map visualizations are generated for the SCrIBE models, and performance metrics are calculated for both models.


## Calculating Sparsity
Here is a code snippet for calculating the sparsity of an image set:
```python
import numpy as np

def calc_sparsity(attn_img):
    attn_img = np.round(attn_img)
    attn_img = attn_img.flatten()
    sparsity = np.count_nonzero(attn_img == 0)
    return sparsity / len(attn_img)

total_sparsity = 0.0

for batch_idx, (data, target) in enumerate(test_loader):

    # Model inference steps...
    
    # For each batch, calculate 'attn' using the 'attention' function provided in the notebooks
    
    for idx in range(len(attn)):
        total_sparsity = total_sparsity + calc_sparsity(attn[idx].numpy().transpose(1,2,0).squeeze())

sparsity = total_sparsity / len(image_set)
```


## Citation
If you find this work useful, please cite our paper:
```tex
@INPROCEEDINGS{Liu2020, 
    author={S. Liu and C. Lehman and G. AlRegib}, 
    booktitle={IEEE International Conference on Image Processing (ICIP)}, 
    title={Robustness and Overfitting Behavior of Implicit Background Models},
    year={2020}
}
```