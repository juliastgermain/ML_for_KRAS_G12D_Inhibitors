# ML_for_KRAS_G12D_Inhibitors
This repository contains the code and datasets for predicting the pIC50 value of FDA approved drugs for the KRAS protein through Machine Learning Models. These were used in "Machine Learning-Driven Drug Repurposing for KRAS G12C and KRAS G12D Inhibition"

The code was authored by:

Bebensee, David <br>
Fuschi, Gianluca <br>
Moawad, Christophe Mina Fahmy <br>
St.Germain, Julia <br>

The outputs of the Extractfeatures files are already available as .csv files so there is no need to rerun any of the code. But if you would want to, here is how to: 
Run the files in the folder 'Extract_features'. This code generates the features for each of the dataset. The ouput is a csv file that looks like this '<molecule>_training.csv'.
The three outputs are all stored in the Raw_files folder. 

To produce the list of top 10 molecules per model for each dataset, here is how it is setup:
The NN folder contains the files to run a prediction with the Neural Network for the respective datasets. 
The RF folder contains separate files for training + testing and prediction with the Random Forest for the respective datasets. 
Each prediction file will output a csv file with the top 10 molecules with the highest pIC50 value.



