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
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np




DF = pd.read_csv("merged_features_IC50_g12c_169.csv")
#DF = DF.dropna()
print(len(DF))

DF = DF.drop_duplicates(subset=['Smiles', 'IC50 (nM)'])


DF = DF.dropna()
print(len(DF))


def pIC50(input_df):
    input_df = input_df.copy()
    input_df["IC50 (nM)"] = pd.to_numeric(input_df["IC50 (nM)"], errors='coerce')

    # Replace zeros with a small value (1e-12 nM = 1e-21 M)
    molar = np.where(input_df["IC50 (nM)"] == 0,
                     1e-12 * 1e-9,
                     input_df["IC50 (nM)"] * 1e-9)

    return -np.log10(molar)
# Filter and sample data before splitting
DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]

#DF['IC50 (nM)'] = DF['IC50 (nM)'].apply(parse_ic50)
DF = DF.dropna(subset=['IC50 (nM)'])  # Remove rows with invalid IC50
print(len(DF))



DF = DF[(DF['FC'] == 0)] #& (DF['IC50 (nM)'] <= 10)]
y = DF['IC50 (nM)']


DF['pIC50'] = pIC50(DF)  # New column
DF = DF[DF['pIC50'] <= 20]
print(len(DF))

y = DF['pIC50']  # <-- Now using correct column
X = DF.drop(columns=["ChEMBL ID", "FC", 'IC50 (nM)', "Smiles", "pIC50"])  # Drop old IC50 and new pIC50


from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Get initial feature names
feature_names = X.columns.tolist()

# 2. Step 1: Variance Threshold (track kept features)
var_selector = VarianceThreshold(threshold=0.8*(1-0.8))
X_var = var_selector.fit_transform(X)
var_mask = var_selector.get_support()
remaining_features = [feature_names[i] for i in range(len(feature_names)) if var_mask[i]]
print(f"After VarianceThreshold: {len(remaining_features)} features")

# 3. Step 2: Univariate Selection (only on remaining features)
X_filtered = pd.DataFrame(X_var, columns=remaining_features)
univariate_selector = SelectKBest(score_func=f_regression, k=min(50, X_filtered.shape[1]))
X_selected = univariate_selector.fit_transform(X_filtered, y)
uni_mask = univariate_selector.get_support()
selected_features = [remaining_features[i] for i in range(len(remaining_features)) if uni_mask[i]]
print(f"After SelectKBest: {len(selected_features)} features")

# 4. Final output
X = pd.DataFrame(X_selected, columns=selected_features)

# After feature selection
joblib.dump(selected_features, 'selected_features.pkl')  # Save feature names

# Scale the data
# Scale X and y properly
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()  # Changed to StandardScaler
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))



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

    def __init__(self, input_dim, hidden_dim, output_dim,

                 num_hidden_layers, dropout_rate):

        super(MultiOutputRegressor, self).__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_hidden_layers - 1):

            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),

                       nn.Dropout(dropout_rate)]

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.model = nn.Sequential(*layers)



    def forward(self, x):

        return self.model(x)



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

    hidden_dim = 256

    num_hidden_layers = 4

    output_dim = y_train_tensor.shape[1]

    dropout_rate = 0.0  # or try 0.2




    model = MultiOutputRegressor(input_dim, hidden_dim, output_dim, num_hidden_layers, dropout_rate)

    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber loss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0041, weight_decay=1e-6)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)



    # Prepare DataLoader

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)



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

model.eval()

with torch.no_grad():

    X_tensor = torch.from_numpy(X_scaled).float()

    y_pred = model(X_tensor)



# Calculate residuals

y_pred_all = y_pred.cpu().numpy()

y_actual_all = y_scaled  # use scaled y for consistency

y_pred_test_all = scaler_y.inverse_transform(y_pred_all)  # Inverse scale predictions

y_test_all = scaler_y.inverse_transform(y_actual_all)  # Inverse scale actual values

residuals = (y_test_all - y_pred_test_all)  # Calculate residuals



# Compute final R² and MSE

final_r2 = r2_score(y_test_all, y_pred_test_all, multioutput='variance_weighted')

variance_y = np.var(y_test_all)

MSE = mean_squared_error(y_actual_all, y_pred_all)

NMSE = MSE / variance_y



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

# plt.show()




predicted_values = {}

fda_pred = pd.read_csv("C:/Users/Gianluca/OneDrive/Documenti"
                       "/University/3 II/Project_Y3/Raw "
                       "Files/FDA_Hyb_Features.csv")

chembl_id_column = fda_pred['ChEMBL ID']

smiles_column = fda_pred['Smiles']



# Process FDA features (drop non-feature columns)

# Process FDA features
# 1. Load saved feature names
selected_features = joblib.load('selected_features.pkl')

# 2. Create empty dataframe with correct columns
X_new = pd.DataFrame(columns=selected_features)

# 3. Fill with FDA data where available
for feat in selected_features:
    if feat in fda_pred.columns:
        X_new[feat] = fda_pred[feat]
    else:
        X_new[feat] = 0  # Fill missing with zeros
        print(f"Added missing feature: {feat}")

# 4. Apply scaling ONLY (no feature selection needed)
X_new_scaled = scaler_X.transform(X_new.values)


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


# Filter out NaN values
filtered_values = {k: v for k, v in predicted_values.items() if not pd.isna(v)}

# Sort the filtered values
sorted_values = sorted(filtered_values.items(), key=lambda x: x[1], reverse=True)

# Create DataFrame with top 11 values
molecules_df = pd.DataFrame(sorted_values[0:31], columns=[
    'chembl_id', 'Predicted_Value'])

# Save to CSV
molecules_df.to_csv('NN_molecules_Newfeatures_G12C_30.csv')