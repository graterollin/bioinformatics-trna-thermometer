# Guide
This repo contains work for implementing and improving upon the work done by Cimen et al. in their paper titled 'Building a tRNA thermometer to estimate microbial adaptation to temperature'

## Data
Contains the data that was used to get the phylogeneticallty-informed models to run. Due to size contraints, the original files can be used for running the models couldn't be added to this repo, but can be found in the original author's bitbucket repo: https://bitbucket.org/bucklerlab/cnn_trna_ogt/src/master/

## Scripts 
Contains the scripts developed for both interpreting the results of the CNN classifier model, and also for helping to construct the distance matrix for the phylogenetically-informed models

## Models 
The rest of the folders contain the different models and their scripts/output data. The classification and regression models are based off the original work from the authors above, while the transformer model was developed for the purpose of attempting to improve on their work. 

The classification model output folder also has the results from the experimentation done on hyperparameter finetuning for the random data split. The results can be found in classification_model-Output/RandomSplit/ArchaeaOutput/OptimizedOutput
