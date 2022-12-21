# Forecasting Exchange Rates

The aim of this project is to forcast daily USD/CHF exchange rate thanks to historical data of: 
- past USD/CHF exchange rate.
- CHF/ other currency exchange rate.
- CHF Bonds (Interest rate)
- USD/ other currency exchange rate.
- USD Bonds (Interest rate)

## main.ipynb

You can see all the pipeline of the project in main.jpynb.
The pipeline consitst in: 

### Preprocessing 
In this part we generate the dataset in order to pass it to ours model, after loadign the data and generate it we inspect the features in order to see that all is okay and look how are the distributions of our data. We can choose if we want to use the augmented dataset or the original dataset (by slicing the data matrices)

### Model 
In this section we implement all our models for price and trend prediction. The autoencoder model is trained or simply loaded from the already saved version in Helper/model/model_AutoEncoder.pth

#### Price prediction 
Here we focus on price prediction and we test the baseline models, models with autoencoder, and models with regularised loss. 

#### Trend prediction
Here we focus on trend prediction and we test the baseline models and the models with autoencoder.

### Result 
In this section we can see all ours result from the previus test and analysis. 

## main_for_report_runs.ipynb

This code generates the numerical data present in the report. It repeats what is done in main.ipynb (see above for structure), but the results are obtained from average of 10 runs of each model. Here we test both the original dataset and the augmente dataset.

## Helper
This folder contains all helper .py scripts, exploited for dataset creation, model structures, training, testing and plots. It also gathers the noteworthy model parameters in the "model" folder, including pretrained autoencoders.

## Data
This folder contains all data acquired and datasets employed. The final and main dataset is Data/data_daily/dataset_daily.csv, which contains daily records of all features.

## Plots
Folder to save plots

## Test
Folder for old scripts, not used for final version of the project

