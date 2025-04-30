import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import joblib
import argparse





DF = pd.read_csv('/Users/user/Downloads/Drug Design FInal/FINAL_GIT/Raw Files/new_merged_features_IC50_g12c.csv')
DF = DF.dropna()

n_folds = 5

DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]

# Filter and sample data before splitting

# Transform IC50 to log scale first
DF['IC50 (nM)'] = DF['IC50 (nM)'].str.lstrip('<>').astype(float)
DF = DF[(DF['FC'] == 0) & (DF['IC50 (nM)'] <= 10000)]  # Increased range to 10,000

# Use log transformation for IC50 values
y = np.log10(DF['IC50 (nM)'])
X = DF.drop(columns=["ChEMBL ID", "FC", 'IC50 (nM)', "Smiles"])

# Scale features with RobustScaler (better for wide ranges)
from sklearn.preprocessing import RobustScaler

scaler_X = RobustScaler()
X_scaled = scaler_X.fit_transform(X)


# Custom scaling for log-transformed IC50 values
class LogScaler:
    def __init__(self):
        self.min_ = None
        self.range_ = None

    def fit(self, y):
        self.min_ = y.min()
        self.range_ = y.max() - y.min()

    def transform(self, y):
        return (y - self.min_) / self.range_

    def inverse_transform(self, y):
        return y * self.range_ + self.min_


scaler_y = LogScaler()
scaler_y.fit(y.values.reshape(-1, 1))
y_scaled = scaler_y.transform(y.values.reshape(-1, 1))


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

class IC50Predictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),

            nn.Linear(128, 1),
            nn.Sigmoid()  # Constrains output to (0,1)
        )

        # Careful initialization
        nn.init.xavier_uniform_(self.net[-2].weight, gain=0.1)
        nn.init.constant_(self.net[-2].bias, 0.5)

    def forward(self, x):
        return self.net(x)



for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):

    print(f"\nFold {fold + 1}/{n_folds}")



    # Split data into training and validation sets for the current fold

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]

    y_train, y_val = y_scaled[train_idx], y_scaled[val_idx]



    # Convert to tensors

    X_train_tensor = torch.from_numpy(X_train).float()

    y_train_tensor = torch.from_numpy(y_train).float()

    X_val_tensor = torch.from_numpy(X_val).float()

    y_val_tensor = torch.from_numpy(y_val).float()



    # Model, loss, optimizer, and scheduler settings

    input_dim = X_train_tensor.shape[1]

    model = IC50Predictor(input_dim)


    class LogMSELoss(nn.Module):
        def __init__(self, scaler_y):
            super().__init__()
            self.scaler_y = scaler_y

        def forward(self, pred, target):
            # Calculate inverse transform mathematically without breaking the graph
            pred_log = pred * self.scaler_y.range_ + self.scaler_y.min_
            target_log = target * self.scaler_y.range_ + self.scaler_y.min_

            # Calculate MSE in log space
            return torch.mean((pred_log - target_log) ** 2)


    criterion = LogMSELoss(scaler_y)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=0.0005,  # Reduced learning rate
                                  weight_decay=0.01)  # Increased regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Add gradient clipping
    max_grad_norm = 1.0


    # Prepare DataLoader

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)



    # Training loop for the current fold

    train_losses, val_losses, r2_scores, mse_scores = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0



        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_log = model(X_val_tensor) * scaler_y.range_ + scaler_y.min_
            y_true_log = y_val_tensor * scaler_y.range_ + scaler_y.min_

            val_loss = criterion(y_pred_log, y_true_log)
            r2 = r2_score(y_true_log.cpu().numpy(), y_pred_log.cpu().numpy())
            MSE = mean_squared_error(y_true_log.cpu().numpy(), y_pred_log.cpu().numpy())
            # Inside your validation loop (after calculating r2 and MSE):
            r2_scores.append(r2)
            mse_scores.append(MSE)



        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1}: Train Loss: {epoch_train_loss / len(train_loader):.4f}, Val R2: {r2:.4f}")


    # Store metrics for this fold

    all_train_losses.append(train_losses)

    all_val_losses.append(val_losses)

    all_r2_scores.append(r2_scores)

    all_mse_scores.append(mse_scores)


# After your cross-validation loop, add this diagnostic and fix:

# 1. First verify all metric array lengths
print("\nMetric array lengths before processing:")
print(f"Train losses: {len(all_train_losses)} folds, lengths: {[len(f) for f in all_train_losses]}")
print(f"Val losses: {len(all_val_losses)} folds, lengths: {[len(f) for f in all_val_losses]}")
print(f"R2 scores: {len(all_r2_scores)} folds, lengths: {[len(f) for f in all_r2_scores]}")
print(f"MSE scores: {len(all_mse_scores)} folds, lengths: {[len(f) for f in all_mse_scores]}")

# 2. Find the minimum epoch count completed by all folds
min_epochs = min(
    min(len(f) for f in all_train_losses),
    min(len(f) for f in all_val_losses),
    min(len(f) for f in all_r2_scores),
    min(len(f) for f in all_mse_scores)
)
print(f"\nMinimum epochs completed by all folds: {min_epochs}")

# 3. Trim all metric arrays to this minimum length
all_train_losses = [fold[:min_epochs] for fold in all_train_losses]
all_val_losses = [fold[:min_epochs] for fold in all_val_losses]
all_r2_scores = [fold[:min_epochs] for fold in all_r2_scores]
all_mse_scores = [fold[:min_epochs] for fold in all_mse_scores]

# 4. Calculate averages
average_train_losses = np.mean(all_train_losses, axis=0)
average_val_losses = np.mean(all_val_losses, axis=0)
average_r2_scores = np.mean(all_r2_scores, axis=0)
average_mse_scores = np.mean(all_mse_scores, axis=0)

# 5. Create DataFrame with consistent lengths
loss_data = pd.DataFrame({
    'Epoch': range(1, min_epochs + 1),
    'Avg_Train_Loss': average_train_losses,
    'Avg_Val_Loss': average_val_losses,
    'Avg_R2_Score': average_r2_scores,
    'Avg_MSE': average_mse_scores
})

print("\nDataFrame creation successful!")
print(loss_data.head())



print("Training completed. Exporting model for visualization...")



# Save the final model (you might want to save the best model based on validation scores)

# torch.save(model.state_dict(), f"best_model_{chembel_id}_SV.pth")

# print(f"Model saved as 'best_model_{chembel_id}_SV.pth'.")




def safe_predict(model, X, min_ic50=1.0, max_ic50=10000.0):
    model.eval()
    with torch.no_grad():
        scaled_pred = model(X)
        # Convert back from log scale
        pred_log = scaler_y.inverse_transform(scaled_pred.cpu().numpy())
        pred_linear = 10**pred_log
        # Clip to physical range
        return np.clip(pred_linear, min_ic50, max_ic50).flatten()

# For validation during training:
val_pred_linear = safe_predict(model, X_val_tensor)
val_true_linear = 10**scaler_y.inverse_transform(y_val_tensor.cpu().numpy())

# For final evaluation:
final_preds = safe_predict(model, torch.from_numpy(X_scaled).float())
final_true = 10**scaler_y.inverse_transform(y_scaled)

# Calculate metrics on linear scale
final_r2 = r2_score(final_true, final_preds)
final_mse = mean_squared_error(final_true, final_preds)
print(f"Final R2: {final_r2:.4f}, Final MSE: {final_mse:.4f}")

# For FDA predictions:
X_new_scaled = scaler_X.transform(X_new)
X_new_tensor = torch.from_numpy(X_new_scaled).float()
fda_predictions = safe_predict(model, X_new_tensor)

# Save predictions
predictions = pd.DataFrame({
    'ChEMBL ID': fda_pred['ChEMBL ID'],
    'Predicted_IC50': fda_predictions
}).sort_values('Predicted_IC50')
predictions.to_csv('NN_molecules_Newfeatures_G12C.csv', index=False)



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




# Update your plotting code to use the safe predictions
plt.figure(figsize=(8, 6))
plt.scatter(final_true, final_preds, alpha=0.6)
plt.plot([final_true.min(), final_true.max()],
         [final_true.min(), final_true.max()], 'r--')
plt.xlabel("Actual IC50 (nM)")
plt.ylabel("Predicted IC50 (nM)")
plt.title(f"IC50 Predictions (R2: {final_r2:.3f})")
plt.tight_layout()
plt.savefig("Actual_vs_Predicted_G12C_SV.png", dpi=300)
plt.show()




predicted_values = {}

fda_pred = pd.read_csv("/Users/user/Downloads/Drug Design FInal/FINAL_GIT/Raw Files/FDA_Hyb_Features.csv")

chembl_id_column = fda_pred['ChEMBL ID']

smiles_column = fda_pred['Smiles']



# Process FDA features (drop non-feature columns)

X_new = fda_pred.drop(columns=["FC", "Smiles", "ChEMBL ID"],

                      errors='ignore')

# Ensure X_new has all columns in X_train and in the correct order

number_of_missing_features = 0

for col in X.columns:

    if col not in X_new.columns:

        number_of_missing_features += 1

        X_new[col] = 0  # Add missing columns with zeros

print(number_of_missing_features, "missing features added to X_new")

X_new = X_new[X.columns]  # Reorder columns to match X_train

# Scale new FDA data

X_new_scaled = scaler_X.transform(X_new)

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



sorted_values = sorted(predicted_values.items(), key=lambda x: x[1])

molecules_df = pd.DataFrame(sorted_values,

                            columns=['chembl_id', 'Predicted_Value'])





molecules_df.to_csv('NN_molecules_Newfeatures_G12C.csv')