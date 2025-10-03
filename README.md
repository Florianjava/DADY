# DADY
DADY Project stands for : "Deep Learning for spatiotemporal and multispectral drone data analysis in agroecological systems to improve trait prediction across genotypes and environments"

##This repository contains the first elements of the project, mainly : 
- a way to access the multiple use-cases spatio-temporal patches (Xt,Yt) for training, in a generic way
- a unet trained inference missing channels to test these features

##Associated data :
* This package can be tested with the use case data of the DADY project, stored on Muse / Phenodrone, with CIRAD access

##Content :
* data_preparation contains scripts for extracting patches from the data, being random and squared, or defined by associated shapefiles. Each USe case is associated with custom scripts, that focus on this specific case, and release functions / accessors dedicated to it.

* data_example contains example patches that have been extracted from the use-cases datasets, using the routines describes in data_preparation scripts

* trained_models : is intended to contain the trained or downloaded neural network models. When running the inference.py, the script looks if a model is present. If not, it downloads it dynamically from hugging face. In order to avoid repetitive models, please download it on your computer. The models can be found on hugging face at the following links :
- trained_models/unet/... put here the last version of the models that you will find at this link : 
- other models to come

* unet_model : scripts describing the architecture of the unet used to predict missing channels of a random multispectral image. Contains the dataloader used for the training


