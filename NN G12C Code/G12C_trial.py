import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import smogn
import joblib
import argparse
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt






DF = pd.read_csv("/Users/user/PycharmProjects/Drug Design FInal/FINAL_GIT/Raw Files/merged_features_IC50_g12c.csv")
#DF = DF.dropna()


print(len(DF))

def pIC50(input):
    pIC50 = []

    input["IC50 (nM)"] = pd.to_numeric(input["IC50 (nM)"],errors='coerce')

    for i in input["IC50 (nM)"]:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['IC50 (nM)'] = pIC50
    x = input["IC50 (nM)"]

    return x
# Filter and sample data before splitting
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
import random

def parse_ic50(val):
    if isinstance(val, str):
        if '<' in val:
            upper = float(val.replace('<', ''))
            return random.uniform(upper / 10, upper)
        elif '>' in val:
            lower = float(val.replace('>', ''))
            return random.uniform(lower, lower * 10)
        else:
            return float(val)
    return val


DF['IC50 (nM)'] = DF['IC50 (nM)'].apply(parse_ic50)
# Before augmentation
DF = DF.drop_duplicates(subset=['Smiles', 'IC50 (nM)'])  # Remove exact duplicates

DF = DF.dropna()
print(len(DF))

DF = DF[(DF['FC'] == 0)] #& (DF['IC50 (nM)'] <= 1)]
non_feature_cols = ['ChEMBL ID', 'Smiles']
DF_non_features = DF[non_feature_cols].copy()

# Process features and target
DF_features = DF.drop(columns=non_feature_cols)
DF_features['pIC50'] = pIC50(DF_features)

# Apply SMOGN only to features
df_augmented = smogn.smoter(
    data=DF_features,
    y='pIC50',
    k=5,
    pert=0.2,
    samp_method='extreme',
    rel_thres=0.5
)


DF_non_features = DF_non_features.reset_index(drop=True)
df_augmented = df_augmented.reset_index(drop=True)
DF = pd.concat([df_augmented, DF_non_features], axis=1)


df_augmented['pIC50'].hist(bins=50)
plt.title("Distribution After SMOGN")
plt.show()

print(f"Data size after SMOGN: {len(DF)}")

# Reassign y from augmented data
y = DF['pIC50']
# insert the newly generated X
X = DF.drop(columns=['IC50 (nM)', 'ChEMBL ID', 'Smiles','pIC50'])# Drop old IC50 and new pIC50
# Add this before cross-validation

from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()
X = selector.fit_transform(X)
X = pd.DataFrame(X)

scaler_X = StandardScaler()
X_scaled_data = scaler_X.fit_transform(X)  # Actual scaled data

scaler_y = MinMaxScaler()
y_scaled_data = scaler_y.fit_transform(y.values.reshape(-1, 1))

# joblib.dump(scaler_X, f"scaler_X_{chembel_id}_SV.pkl")

# joblib.dump(scaler_y, f"scaler_y_{chembel_id}_SV.pkl")

n_folds = 5

# K-Fold Cross-Validation setup

num_epochs = 200

kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)



# Store metrics across folds

all_train_losses = []

all_val_losses = []

all_r2_scores = []

all_mse_scores = []



import torch.nn.functional as F

# Define the neural network model

class MultiOutputRegressor(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # Split raw data first
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]
    y_train_raw, y_val_raw = y.iloc[train_idx], y.iloc[val_idx]

    # Scale training data and apply to validation
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train_raw)
    X_val = scaler_X.transform(X_val_raw)

    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1))
    y_val = scaler_y.transform(y_val_raw.values.reshape(-1, 1))

    print(f"\nFold {fold + 1}/{n_folds}")





    # Convert to tensors

    X_train_tensor = torch.from_numpy(X_train).float()

    y_train_tensor = torch.from_numpy(y_train).float()

    X_val_tensor = torch.from_numpy(X_val).float()

    y_val_tensor = torch.from_numpy(y_val).float()



    # Model, loss, optimizer, and scheduler settings

    input_dim = X_train_tensor.shape[1]

    hidden_dim = 256

    num_hidden_layers = 4

    output_dim = y_train_tensor.shape[1]

    dropout_rate = 0.0



    model = MultiOutputRegressor(input_dim)

    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0041, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)



    # Prepare DataLoader

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)



    # Training loop for the current fold

    train_losses, val_losses, r2_scores, mse_scores = [], [], [], []



    for epoch in range(num_epochs):

        # Training

        model.train()

        epoch_train_loss = 0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = criterion(y_pred, y_batch)

            loss.backward()

            optimizer.step()

            epoch_train_loss += loss.item()

        train_losses.append(epoch_train_loss / len(train_loader))



        # Validation

        model.eval()

        epoch_val_loss = 0

        y_pred_val_all, y_val_all = [], []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch)

                epoch_val_loss += loss.item()

                y_pred_val_all.append(y_pred)

                y_val_all.append(y_batch)

        val_losses.append(epoch_val_loss / len(val_loader))



        # Calculate R2 score

        y_pred_val_all = torch.cat(y_pred_val_all, dim=0).cpu().numpy()

        y_val_all = torch.cat(y_val_all, dim=0).cpu().numpy()

        r2 = r2_score(y_val_all, y_pred_val_all, multioutput='variance_weighted')

        MSE = mean_squared_error(y_val_all, y_pred_val_all)

        r2_scores.append(r2)

        mse_scores.append(MSE)



        scheduler.step()



        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, R2: {r2:.4f}")



    # Store metrics for this fold

    all_train_losses.append(train_losses)

    all_val_losses.append(val_losses)

    all_r2_scores.append(r2_scores)

    all_mse_scores.append(mse_scores)

    # Save scalers from the last fold
    final_scaler_X = scaler_X
    final_scaler_y = scaler_y

# Aggregate metrics across folds

average_train_losses = np.mean(all_train_losses, axis=0)

average_val_losses = np.mean(all_val_losses, axis=0)

average_r2_scores = np.mean(all_r2_scores, axis=0)

avarage_mse_scores = np.mean(all_mse_scores, axis=0)

print(f'final r2: {average_r2_scores[-1]} +/- {np.std(all_r2_scores, axis=0)[-1]}')

print(f'final mse: {avarage_mse_scores[-1]} +/- {np.std(all_mse_scores, axis=0)[-1]}')

# Save final metrics

loss_data = pd.DataFrame({

    'Epoch': range(1, num_epochs + 1),

    'Avg_Train_Loss': average_train_losses,

    'Avg_Val_Loss': average_val_losses,

    'Avg_R2_Score': average_r2_scores,

    'Avg_MSE': avarage_mse_scores,

})



print("Training completed. Exporting model for visualization...")



# Save the final model (you might want to save the best model based on validation scores)

# torch.save(model.state_dict(), f"best_model_{chembel_id}_SV.pth")

# print(f"Model saved as 'best_model_{chembel_id}_SV.pth'.")





# Inference on the entire dataset

# Inference section - CORRECTED
model.eval()
with torch.no_grad():
    # Use the last fold's scaler
    X_scaled_inference = final_scaler_X.transform(X)
    X_tensor = torch.from_numpy(X_scaled_inference).float()
    y_pred = model(X_tensor)

# Inverse transform predictions with last fold's scaler
y_pred_test_all = final_scaler_y.inverse_transform(y_pred.cpu().numpy())
y_test_all = y.values.reshape(-1, 1)  # Original values, no scaling

# Calculate residuals
residuals = y_test_all - y_pred_test_all



# Compute final R² and MSE

# Compute metrics - CORRECTED
final_r2 = r2_score(y_test_all, y_pred_test_all)
MSE = mean_squared_error(y_test_all, y_pred_test_all)
NMSE = MSE / np.var(y_test_all)




print(f"Final R2 Score: {final_r2:.4f}")

print(f"MSE: {MSE:.4f}")

print(f"NMSE: {NMSE:.4f}")



# Save performance metrics to loss data

loss_data['Final_R2'] = final_r2

loss_data['MSE'] = MSE

loss_data['NMSE'] = NMSE



# Save the loss data including the final metrics

# loss_data.to_csv(f"loss_and_r2_data_{chembel_id}_SV.csv", index=False)

#

# # Save predictions and actual values for further analysis

# np.savez(f"predictions_vs_actuals_{chembel_id}_SV.npz",

#          y_pred=y_pred_test_all,

#          y_test=y_test_all,

#          residuals=residuals)



print("Inference completed and results saved.")



import matplotlib.pyplot as plt

import matplotlib as mpl

from matplotlib.ticker import FuncFormatter

from matplotlib.ticker import FormatStrFormatter

# Global settings for consistent styling

mpl.rcParams['lines.linewidth'] = 3

mpl.rcParams['legend.fontsize'] = 18

mpl.rcParams['axes.labelsize'] = 18

mpl.rcParams['xtick.labelsize'] = 16

mpl.rcParams['ytick.labelsize'] = 16



def clean_plot(ax, legend=True):

    """Custom function to clean plots."""

    ax.spines['right'].set_visible(False)

    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_linewidth(3)    # Y-axis

    ax.spines['bottom'].set_linewidth(3)  # X-axis

    if legend:

        ax.legend(frameon=False, fontsize=16)



# Function to format ticks to scale of 10^3

def scale_to_thousands(x, pos):

    return f'{x/1e3:.0f}'





formatter = FuncFormatter(scale_to_thousands)



plt.figure(figsize=(5, 5))



plt.scatter(y_test_all, y_pred_test_all, alpha=0.6, label='SV')

plt.plot([y_test_all.min(), y_test_all.max()],

         [y_test_all.min(), y_test_all.max()], 'r--', label='line of equality')



plt.xlabel("Actual Values")# (x $10^3$)")

plt.ylabel("Predicted Values")# (x $10^3$)")

plt.legend()

# plt.gca().xaxis.set_major_formatter(formatter)

# plt.gca().yaxis.set_major_formatter(formatter)

clean_plot(plt.gca())

plt.subplots_adjust(bottom=0.2)

plt.tight_layout()

plt.savefig(f"Actual_vs_Predicted_G12C_SV.png", dpi=300)

plt.show()




predicted_values = {}

fda_pred = pd.read_csv("/Users/user/PycharmProjects/Drug Design FInal/FINAL_GIT/Raw Files/FDA_Hyb_Features.csv")

chembl_id_column = fda_pred['ChEMBL ID']

smiles_column = fda_pred['Smiles']



# Process FDA features (drop non-feature columns)

X_new = fda_pred.drop(columns=["FC", "Smiles", "ChEMBL ID"],

                      errors='ignore')

X_new = selector.transform(X_new)
X_new = pd.DataFrame(X_new)

# Ensure X_new has all columns in X_train and in the correct order

number_of_missing_features = 0

for col in X.columns:

    if col not in X_new.columns:

        number_of_missing_features += 1

        X_new[col] = 0  # Add missing columns with zeros

print(number_of_missing_features, "missing features added to X_new")

X_new = X_new[X.columns]  # Reorder columns to match X_train

# Scale new FDA data

# Use the final scaler (from last CV fold) for new predictions
X_new_scaled = final_scaler_X.transform(X_new)  # Not scaler_X

# Convert to PyTorch tensor

X_new_tensor = torch.from_numpy(X_new_scaled).float()



# Predict new values

model.eval()

with torch.no_grad():

    y_new_pred = model(X_new_tensor)



# Convert predictions back to original pIC50 scale

y_new_pred_original = scaler_y.inverse_transform(

    y_new_pred.numpy()).flatten()



# Store predictions

for chembl_id, predicted_value in zip(chembl_id_column,

                                      y_new_pred_original):

    if chembl_id not in predicted_values:

        predicted_values[chembl_id] = []

    predicted_values[chembl_id].append(predicted_value)



# print(predicted_values)



sorted_values = sorted(predicted_values.items(), key=lambda x: x[1], reverse= True)

molecules_df = pd.DataFrame(sorted_values[0:11],

                            columns=['chembl_id', 'Predicted_Value'])





molecules_df.to_csv('NN_molecules_Newfeatures_G12C(1).csv')