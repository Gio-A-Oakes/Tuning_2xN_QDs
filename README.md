## Systematically tuning a 2×N array of quantum dots with machine learning:

Implementation of an upcoming paper on:

> Silicon quantum dots (QDs) are a compelling platform for a fault-tolerant quantum computer due to its scalability, high coherence times and easy integration into current electronics and fabrication lines.
> However, due to inter-dot coupling, it is difficult to control each QD independently, resulting in a complex calibration process, which becomes impossible to heuristically tune as the number of qubits increases towards a NISQ device.
> Inspired by recent demonstrations of scalable quantum dot arrays, we develop a framework to tune a 2×N array of QDs, based off gradients of different transition lines that can be measured in multiple two-dimensional stability diagrams.
> To automate the process, we successfully train a neural network to extract the gradients from a Hough transformation of the stability diagram of a double QD and test the algorithm on simulated data of a 2×2 QD array.

## Simulating a 2xN array of QDs:

The Mosquito project has been making silicon QD arrays by modifying current transistor technology as least as possible, resulting in a 2xN architecture. As a result, the 'random_c' function takes this geometry into account, as the parameter *ratio* is used to determine the relationship between the mutual and cross capacitances perpendicular, parallel or diagonally across the nanowire. Once the 

## Automatically extracting the gradients of a DQD:

## Number of measurements required:
