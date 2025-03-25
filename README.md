# ML_for_KRAS_G12D_Inhibitors
This repository contains the code and datasets for predicting the pIC50 value of FDA approved drugs for the KRAS protein through Machine Learning Models. These were used in "insert article name here! and add DOI"

The code was authored by:

Bebensee, David <p>
Fuschi, Gianluca <p>
Moawad, Christophe Mina Fahmy <p>
St.Germain, Julia <p>

Instructions on use
All of the outputs are already available as .csv files so there is no need to rerun any of the code. But if you would want to, here is the order and what each file does:

Run the data_fetch_ChEMBL_KRAS.ipnyb file to get the training_descriptors.csv. This code accesses a ChEMBL webclient and saves all molecules that have a recorded IC50 with the GTPase KRas molecule (CHEMBL2189121). Then using the RDkit Library we generate 209 descriptors from the SMILES code of each molecule. Before being exported the IC50 are being standardized to pIC50. Muhamed's file already has the new and old descriptors, so we need to ask what code he used to generate the features like we did in the step above.

Run the removing_outliers.ipnyb and remove the outliers from the training_descriptors.csv and output training_descriptors_no_outliers.csv This code runs for 50 iterations and identifies outliers that occur multiple times. The top 8 outliers will then be removed from the dataset. We need to ask Muhamed if he also removed outliers from his file already or not.

Run the transform_fda.ipnyb file on the fda.csv file which we got from the ChEMBL databast by filtering approved and 1 or less Rule of 5 violations, you will end up with the FDA_features.csv. This code runs the same function that generates 209 descriptors with RDkit as data_fetch_ChEMBL_KRAS.ipnyb We use the lateest fda approved drugs normally, but then we use Muhamed's new code of extract features to generate the file fda_original_Hyb_Fetures3.csv

Run the prediction_code.ipnyb file with the FDA_features.csv and training_descriptors_no_outliers.csv, you will receive the all_results.csv. This code outputs a file which contains the best 13 molecules that appeared over 50 iterations of runs. Values are IC50 so we use the code of pIC50 to change the values. Then we run the prediction code with Muhamed"s file for training and predict for fda_original_Hyb_Fetures3.csv

maybe see how to change the code to submit more streamlined results.
