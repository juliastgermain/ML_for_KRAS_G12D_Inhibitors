
# Machine Learning for KRAS G12C/D Drug Discovery

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-published-success.svg)
![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-orange.svg)

> **Machine learning-driven drug repurposing approach identifying FDA-approved compounds as potential KRAS G12C and G12D inhibitors**

📄 **Published preprint**: [bioRxiv (2025)](https://www.biorxiv.org/cgi/content/short/2025.05.16.654410v1)

## 🎯 Project Overview

KRAS mutations (G12C and G12D) are among the most common oncogenic drivers in human cancers, making them critical therapeutic targets. This project applies machine learning to identify potential KRAS inhibitors from FDA-approved drugs, enabling rapid drug repurposing.

We developed **Random Forest** and **Neural Network** models to predict drug activity (pIC50) against KRAS G12C/D mutations using molecular features, achieving strong predictive performance and identifying promising drug candidates.

## 🔬 Key Results

- **Identified 3 FDA-approved drugs** as potential KRAS inhibitors:
  - **Cobimetinib** (MEK inhibitor)
  - **Gilteritinib** (FLT3 inhibitor)
  - **Acalabrutinib** (BTK inhibitor)
  
- **Feature importance analysis** revealed key molecular descriptors influencing drug activity
- **Coordinated with wet-lab collaborators** for experimental validation of predictions
- **Models trained on molecular descriptors** extracted from chemical structures using RDKit

## 🛠️ Technologies Used

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Random Forest), TensorFlow/Keras (Neural Networks)
- **Cheminformatics**: RDKit (molecular descriptor calculation)
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn

## 📊 Project Structure
```
ML_for_KRAS_G12D_Inhibitors/
├── Extract_Features/              # Feature extraction from molecular structures
│   ├── G12C_Extractfeatures.ipynb
│   ├── G12D_Extractfeatures.ipynb
│   ├── GTPase_KRas_Extractfeatures.ipynb
│   └── Hybrid.py
├── NN/                            # Neural Network models
│   ├── NN_Train_Test_Prediction_G12C.ipynb
│   ├── NN_Train_Test_Prediction_G12D.ipynb
│   └── NN_Train_Test_Prediction_GTPase.ipynb
├── RF/                            # Random Forest models
│   ├── RF_Train_Test_G12C.ipynb           # Training & testing
│   ├── RF_Train_Test_G12D.ipynb
│   ├── RF_Train_Test_GTPase_KRas.ipynb
│   ├── RF_Prediction_G12C.ipynb           # Predictions on FDA drugs
│   ├── RF_Prediction_G12D.ipynb
│   └── RF_Prediction_GTPase_KRas.ipynb
├── Raw_Files/                     # Pre-computed molecular features (CSV)
│   ├── G12C_training.csv
│   ├── G12D_training.csv
│   └── FDA_approved_drugs.csv
├── Quality_of_Life/               # Utility scripts and helpers
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/juliastgermain/ML_for_KRAS_G12D_Inhibitors.git
cd ML_for_KRAS_G12D_Inhibitors

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Option 1: Use Pre-computed Features (Recommended)

Pre-extracted molecular features are available in `Raw_files/`. Skip directly to model training.

#### Option 2: Extract Features from Scratch
```bash
# Extract features for each dataset
python Extract_features/G12C_extract.py
python Extract_features/G12D_extract.py
python Extract_features/FDA_extract.py
```

This generates `*_training.csv` files in the `Raw_files/` folder.

#### Running Predictions

**Random Forest Models:**
```bash
# Train and test
python RF/G12C_RF_train_test.py
python RF/G12D_RF_train_test.py

# Predict on FDA-approved drugs
python RF/G12C_RF_predict.py  # Outputs top 10 candidates
python RF/G12D_RF_predict.py
```

**Neural Network Models:**
```bash
# Run predictions
python NN/G12C_NN.py  # Outputs top 10 candidates
python NN/G12D_NN.py
python NN/FDA_NN.py
```

Each prediction script outputs a CSV file with the **top 10 molecules** ranked by predicted pIC50 value.

## 🧬 Molecular Features

Features extracted using RDKit include:
- Molecular weight
- LogP (lipophilicity)
- Number of hydrogen bond donors/acceptors
- Topological polar surface area (TPSA)
- Molecular fingerprints

## 🔮 Future Work

- Experimental validation of top predicted compounds
- Expand to other KRAS mutation variants (G13D, Q61H)
- Incorporate 3D molecular structure information
- Explore deep learning architectures (Graph Neural Networks)
- Validate predictions with molecular docking simulations

## 👥 Authors

The code was authored by:

Bebensee, David <br>
Fuschi, Gianluca <br>
Moawad, Christophe Mina Fahmy <br>
St.Germain, Julia <br>
Mohamed, Ashraf <br>
Amin, Muhamed <br>


