from rdkit import Chem
from rdkit.Chem import AllChem

def polFeature(molecule):
    #elements = ['Se','Si']
    H = 0
    Cl = 0
    Br = 0
    I = 0
    P = 0
    F = 0
    Se = 0
    Si = 0
    S_sp3 = 0
    S_sp2 = 0
    S_sp  = 0
    C_sp3 = 0 
    C_sp2 = 0
    C_sp  = 0
    N_sp3 = 0
    N_sp2 = 0
    N_sp  = 0
    O_sp3 = 0
    O_sp2 = 0
    O_sp  = 0
    FC = 0
    N = 0
    #molecule = Chem.MolFromSmiles(smiles)
    #molecule = Chem.AddHs(molecule)
    atms = molecule.GetAtoms()
    for i in atms:
        elem = i.GetSymbol()
        #if elem in elements:return None
        if elem == "H":
            H+=1
        elif elem == "Cl":
            Cl += 1
        elif elem == "Br":
            Br += 1
        elif elem == "I":
            I += 1
        elif elem == "P":
            P += 1
        elif elem == "F":
            F += 1
        if elem=="C":
            hyprd = str(i.GetHybridization())
            if hyprd == "SP":C_sp+=1
            if hyprd == "SP2":C_sp2+=1
            if hyprd == "SP3":C_sp3+=1
        elif elem=="N":
            hyprd = str(i.GetHybridization())
            if hyprd == "SP":N_sp+=1
            if hyprd == "SP2":N_sp2+=1
            if hyprd == "SP3":N_sp3+=1
        elif elem=="O":
            hyprd = str(i.GetHybridization())
            if hyprd == "SP":O_sp+=1
            if hyprd == "SP2":O_sp2+=1
            if hyprd == "SP3":O_sp3+=1
        elif elem=="S":
            hyprd = str(i.GetHybridization())
            if hyprd == "SP":S_sp+=1
            if hyprd == "SP2":S_sp2+=1
            if hyprd == "SP3":S_sp3+=1
        N+=i.GetAtomicNum()
        FC+=i.GetFormalCharge()
    return [FC,N,H,Cl,Br,I,P,F,Se,Si,S_sp3,S_sp2,S_sp,C_sp3,C_sp2,C_sp,N_sp3,N_sp2,N_sp,O_sp3,O_sp2,O_sp]

