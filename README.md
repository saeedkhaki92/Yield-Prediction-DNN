# Yield-Prediction-DNN


This repository contains my code for the "Crop Yield Prediction Using Deep Neural Networks" paper. The network is a deep feedforward neural network which uses state-of-the-art deep learning techniques such as residual learning, batch normalization, dropout, L1 and L2 regularization.


## Getting Started 

 Please install following packages in Python3:
 
 
 - numpy 
 - tensorflow
 - matplotlib
 
 
 ## Data
 
 - Genotype , soil, and weather data were used in the paper. You should load your data as train and test, then run the model.
 
 - The genotype data were coded in `-1`, `0`, `1` values, respectively representing aa, aA, and AA alleles. The Genotype data has dimension `n-by-p` where n and p denote the number of obseration and genetic markers. 
 
 - The environment data (weather and soil) has dimension `n-by-k`, where n and k denote the number of obseration and enviromental components. 
 
 
 - Each observation is fed to the network as 1d vectors (dimension p+k) .
 

