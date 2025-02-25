# SpectraZones - Deep Autoencoders and Self Organizing Maps Applied to HSI Clustering

Code repository for
**Segmentation of Lithotypes and Hydrothermal Alteration Zones using Deep Autoencoders and Self Organizing Maps Applied to Drill-core Hyperspectral Data: A Case Study at the Prominent Hill IOCG Deposit, Australia**
with implementation of HSI clustering algorithm based on Deep Autoencoders (DAE), Self Organizing Maps (SOM), and agglomerative clustering.

>[Link to article]()

>[DOI - https://doi.org/]()


## Authors
[Elias M. G. Prado](mailto:elias.prado@sgb.gov.br), [ResearchGate](https://www.researchgate.net/profile/Elias_Prado3)<sup>a,b</sup>  
[Carlos Roberto de Souza Filho](https://portal.ige.unicamp.br/en/faculty/carlos-roberto-de-souza-filho)<sup>a</sup>  

<sup>a</sup>*Institute of Geosciences, University of Campinas (UNICAMP), Campinas, São Paulo, Brazil*  
<sup>b</sup>*Centre for Applied Geosciences (CGA), Geological Survey of Brazil (SBG/CPRM), Brasília, Distrito Federal, Brazil*  

## Instructions
### Enviroment Setup

#### Python and Libraries Versions

    Python==3.11.11

    pytorch==2.5.1
    geopandas==0.14.2
    contextily==1.6.2
    plotly==6.0.0
    pysptools==0.15.0
    quicksom==1.0.0
    seaborn==0.13.2
    tqdm==4.67.1
    joblib==1.4.2

#### Download and install Anaconda

Download and install Anaconda form [here](https://www.anaconda.com/).

#### Create conda enviroment

On CMD prompt run the following command:

    conda env create --prefix=./env python=3.11.* jupyter notebook

#### Install required libraries

On CMD prompt run the following command:

    # Install Pytorch
    # Follow instructions on pytorch site for correclty pytorch installation
    # https://pytorch.org/get-started/locally/
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

    # Install Geopandas
    conda install geopandas

    # Install pip libraries
    pip install contextily plotly pysptools quicksom seaborn tqdm joblib


#### Check Pytorch installation (only for CUDA)

On CMD prompt run the following command:

    python
    >> import torch
    >> torch.cuda.is_available() # must return true
    
### Code Organization

The implementation is organized in 5 jupyter notebooks:

- 1 - DataPrep.ipynb
    - Dataset preparation for training the DAE
- 2 - DAE.ipynb
    - DAE training
- 3 - SOM.ipynb
    - SOM training and agglomerative clustering
- 4 - ClusterSegmentation.ipynb
    - Segmentation of Clusters
- 5 - PlotResults.ipynb
    - Plots with the clustering results