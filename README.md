## Automatic virtual voltage extraction of a 2×N array of quantum dots with machine learning:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4739264.svg)](https://doi.org/10.5281/zenodo.4739264)

Implementation of the paper:

[arxiv preprint](https://arxiv.org/abs/2012.03685)


Due to the size of the regression models, they can be found at [this Google drive folder](https://drive.google.com/drive/folders/1amMVMkTsCeIiJBEwIU3OSW-6XexxewId)
The experimental data instead can be downloaded from [here:](https://drive.google.com/drive/folders/1wmCYZmD6UKZPqLdywTVCbtSW9JxUXN_3)

## Simulating a 2xN array of QDs:

The Mosquito project has been making silicon QD arrays by modifying current transistor technology as least as possible, resulting in a 2xN architecture. As a result, the `random_c` function takes this geometry into account, as the parameter *ratio* is used to determine the relationship between the mutual and cross capacitances perpendicular, parallel or diagonally across the nanowire. The matrix *CC* has the gate capacitors of each QD on the main diagonal and the cross capacitors off-diagonal, which allows to speed up the calculation via vectorization. The function `stability_diagram` then calculates the corresponding stability diagram taking inspiration from [Wiel, W. G. Van Der. (2003). Electron transport through double quantum dots.](https://arxiv.org/pdf/cond-mat/0205350.pdf)

For a more detailed example, see the [Jupyter notebook](https://github.com/Gio-A-Oakes/Tuning_2xN_QDs/blob/master/Code/Simulating_2xN_array_of_QDs.ipynb) 

<p align="center">
  <img src="https://github.com/Gio-A-Oakes/Tuning_DQD/blob/master/Figures/Device.png" width="500">
</p>


## Automatically extracting the gradients of a DQD:

We have developed an algorithm designed to extract the virtual voltages from a stability diagram when applying voltages along two of the gates, without human intervention. This is done by using a Hough transform on thresholded data to obtain a histogram of the best fit $\theta$ values. However, due to the presence of more peaks than expected and experimental noise, it is relatively hard to interpret. As a result, a neural network has been trained to extract the required gradients on theoretical data. An outline of the different steps is highlighted below and more information can be found on the [Jupyter notebook](https://github.com/Gio-A-Oakes/Tuning_2xN_QDs/blob/master/Code/Simulating_2xN_array_of_QDs.ipynb)

<p align="center">
  <img src="https://github.com/Gio-A-Oakes/Tuning_DQD/blob/master/Figures/algorithm.png" width="500">
</p>


## Number of measurements required:
To reconstruct the **G** transformation matrix, if only nearest neighbours are taken into account, 5N-4 stability diagrams are required to be measured. We apply two gate voltages at a time in the low electron regime, such that the system behaves similar to a DQD. The algorithm developed to extract the gradients can be easily implemented. As the different measurements are taken, the **G** transformation matrix can be constructed and thus allowing us to change our basis from gate voltage to virtual voltage space as shown for a 2x2 array of QDs:

<p align="center">
  <img src="https://github.com/Gio-A-Oakes/Tuning_DQD/blob/master/Figures/Poster.png" width="500">
</p>
