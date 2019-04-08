# Yield-Prediction-DNN


This repository contains my code for the "Crop Yield Prediction Using Deep Neural Networks" paper.


## Getting Started 

 Please install following packages in Python3:
 
 
 - numpy 
 - tensorflow
 - matplotlib
 
 
 ## Data
 
 - Genotype , soil, and weather data were used in the paper. You should load your data as train and test, then run the model.
 
 - The genotype data were coded in `-1`, `0`, `1` values, respectively representing aa, aA, and AA alleles. The Genotype data is `n p` where $n$ and p denote the number of obseration and genetic markers. 
 - Data are fed to the network as 1d vectors.
