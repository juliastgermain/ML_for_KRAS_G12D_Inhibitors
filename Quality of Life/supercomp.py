import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold

def pIC50(input):
    pIC50 = []

    input["IC50 (nM)"] = pd.to_numeric(input["IC50 (nM)"],errors='raise')

    for i in input["IC50 (nM)"]:
        molar = i*(10**-9) # Converts nM to M
        pIC50.append(-np.log10(molar))

    input['IC50 (nM)'] = pIC50
    x = input["IC50 (nM)"]

    return x




path = "C:\\Users\TheSh\Documents\Programming_in_Python_Class\PyCharmProjects\ML_for_KRAS_G12D_Inhibitors\Raw Files\merged_features_IC50_g12c.csv"
DF = pd.read_csv(path)

DF = DF.loc[:, ~DF.columns.str.contains('^Unnamed')]
DF['IC50 (nM)'] = DF['IC50 (nM)'].str.lstrip('<>').astype(float)

df = DF.dropna()
X = df.drop(columns=['IC50 (nM)','Smiles', 'ChEMBL ID'])

variance = VarianceThreshold()
var_thres = variance.fit_transform(X)

df['pIC50'] = pIC50(df)  # New column
y = df['pIC50']

var_thres = pd.DataFrame(var_thres)

estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit_transform(var_thres, y)
X_full = selector.transform(var_thres)
pd.DataFrame(X_full).to_csv("X_g12c")