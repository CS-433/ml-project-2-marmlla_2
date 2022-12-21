# Forcast Exchange Rate

The haim of this project is to forcast daily USD/CHF exchange rate thanks to historical data of: 
- past USD/CHF exchange rate.
- CHF/ other currency exchange rate.
- CHF Bonds (Interest rate)
- USD/ other currency exchange rate.
- USD Bonds (Interest rate)

## Structure of project

You can see all ours pipeline in main.jpynb.
The pipeline consitst in: 

### Preprocessing 
In this part we generate the dataset in order to pass it to ours model, after load the data and generate it we inspect the feature in order to see that all is okay and look how are the distributions of our data. 

### Model 
In this section we implement all ours model for price and trend prediction. 

#### Price prediction 
Here we focus on price prediction and we test the baseline models, models with autoencoder, and models with regularised loss. 

#### Trend prediction
Here we focus on trend prediction and we test the baseline models and the models with autoencoder.

### Result 
In this section we can see all ours result from the previus test and analysis. 



