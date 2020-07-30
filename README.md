## Systematically tuning a 2×N array of quantum dots with machine learning:

Implementation of an upcoming paper on:

> Silicon quantum dots (QDs) are a compelling platform for a fault-tolerant quantum computer due to its scalability, high coherence times and easy integration into current electronics and fabrication lines.
> However, due to inter-dot coupling, it is difficult to control each QD independently, resulting in a complex calibration process, which becomes impossible to heuristically tune as the number of qubits increases towards a NISQ device.
> Inspired by recent demonstrations of scalable quantum dot arrays, we develop a framework to tune a 2×N array of QDs, based off gradients of different transition lines that can be measured in multiple two-dimensional stability diagrams.
> To automate the process, we successfully train a neural network to extract the gradients from a Hough transformation of the stability diagram of a double QD and test the algorithm on simulated data of a 2×2 QD array.

## Simulating a 2xN array of QDs:

The Mosquito project has been making silicon QD arrays by modifying current transistor technology as least as possible, resulting in a 2xN architecture. As a result, the `random_c` function takes this geometry into account, as the parameter *ratio* is used to determine the relationship between the mutual and cross capacitances perpendicular, parallel or diagonally across the nanowire. The matrix *CC* has the gate capacitors of each QD on the main diagonal and the cross capacitors off-diagonal, which allows to speed up the calculation via vectorization. The function `stability_diagram` then calculates the corresponding stability diagram taking inspiration from [Wiel, W. G. Van Der. (2003). Electron transport through double quantum dots.](https://arxiv.org/pdf/cond-mat/0205350.pdf).

For a more detailed example, see the jupyter notebook 

<p align="center">
  <img src="https://github.com/Gio-A-Oakes/Tuning_DQD/blob/master/Figures/Device.png" width="500">
</p>


## Automatically extracting the gradients of a DQD:

We have developed an algorithem designed to extract the virtual voltages from a DQD stability diagram without human intervention. This is done by using a Hough transform on thresholded data to obtain a histogram of the best fit $\theta$ values. However, due to the presence of more peaks than expected and experimental noise, it is realtivly hard to interpret. As a result, a neural network has been trained to extract the required gradients on theoretical data. An outline of the different steps is highlighted below and more information can be found on the jupyter notebook

<p align="center">
  <img src="https://github.com/Gio-A-Oakes/Tuning_DQD/blob/master/Figures/algorithm.png" width="500">
</p>


## Number of measurements required:

To reconstruct the **G** transformation matrix, if only nearest neighbours are taken into account, 5N-4 stability diagrams are required to be measured. We apply two gate voltages at a time in the low electron regime, such that the system behaves as a DQD. This is because only the QDs that are being probed will be populated and as such, the stability diagram formed is easy to interpret and the algorithem developed to extract the gradients can be easily implemented.
As the different measurments are taken, the **G** transformation matrix can be constructed and thus allowing us to change our basis from gate voltage to virtual voltage space as shown for a 2x2 array of QDs:
<p align="center">
  <img src="https://github.com/Gio-A-Oakes/Tuning_DQD/blob/master/Figures/Poster.png" width="500">
</p>
