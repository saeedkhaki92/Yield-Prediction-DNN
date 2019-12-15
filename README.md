# Yield-Prediction-DNN


This repository contains my code for the "Crop Yield Prediction Using Deep Neural Networks" paper authered by Saeed Khaki and Lizhi Wang. The network is a deep feedforward neural network which uses the state-of-the-art deep learning techniques such as residual learning, batch normalization, dropout, L1 and L2 regularization.

### Please cite our paper if you use our code. Thanks!
```
@article{khaki2019crop,
  title={Crop yield prediction using deep neural networks},
  author={Khaki, Saeed and Wang, Lizhi},
  journal={Frontiers in plant science},
  volume={10},
  year={2019},
  publisher={Frontiers Media SA}
}
```

## Getting Started 

 Please install the following packages in Python3:
 
 
 - numpy 
 - tensorflow
 - matplotlib
 
 
 ## Dimension of Input Data
 
 - Genotype , soil, and weather data were used in the paper. You should load your data as train and test, then run the model.
 
 - The genotype data were coded in `-1`, `0`, `1` values, respectively representing aa, aA, and AA alleles. The genotype data had dimension `n-by-p` where n and p denote the number of obseration and genetic markers. 
 
 - The environment data (weather and soil) had dimension `n-by-k`, where n and k denote the number of obseration and enviromental components. 
 
 
 - Each observation is fed to the network as 1d vectors (dimension p+k) .
 

##  Data Availability Statement 

The data analyzed in this study was provided by Syngenta for 2018 Syngenta Crop Challenge. We accessed
the data through annual Syngenta Crop Challenge. During the challenge, September 2017 to January 2018,
the data was open to the public. Researchers who wish to access the data may do so by contacting Syngenta
directly.

# a

