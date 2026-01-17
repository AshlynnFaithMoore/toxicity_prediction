"""
Molecular Toxicity Prediction for Drug Discovery
This project predicts molecular toxicity using machine learning on chemical structures.
"""


import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


# STEP 2: Load Data


def load_tox21_data():
    """
    Load Tox21 dataset using DeepChem
    Tox21 is a dataset for predicting toxicity of compounds
    """
    try:
        import deepchem as dc
        
        # Load Tox21 dataset
        print("Loading Tox21 dataset...")
        tox21_tasks, tox21_datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
        
        train_dataset, valid_dataset, test_dataset = tox21_datasets
        
        # Extract SMILES and labels for the first toxicity task
        train_smiles = [mol.get_smi() for mol in train_dataset.ids]
        train_labels = train_dataset.y[:, 0]  # First toxicity assay
        
        test_smiles = [mol.get_smi() for mol in test_dataset.ids]
        test_labels = test_dataset.y[:, 0]
        
        # Remove NaN labels
        train_mask = ~np.isnan(train_labels)
        test_mask = ~np.isnan(test_labels)
        
        train_smiles = [s for s, m in zip(train_smiles, train_mask) if m]
        train_labels = train_labels[train_mask]
        
        test_smiles = [s for s, m in zip(test_smiles, test_mask) if m]
        test_labels = test_labels[test_mask]
        
        print(f"Loaded {len(train_smiles)} training samples, {len(test_smiles)} test samples")
        
        return train_smiles, train_labels, test_smiles, test_labels
        
    except ImportError:
        print("DeepChem not installed. Using sample data...")
        return create_sample_data()

def create_sample_data():
    """
    Create sample molecular data for demonstration
    """
    # Sample SMILES strings (real molecules)
    sample_molecules = [
        "CC(C)Cc1ccc(cc1)C(C)C(O)=O",  # Ibuprofen
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "c1ccccc1",  # Benzene (toxic)
        "CCO",  # Ethanol
        "C1=CC=C(C=C1)O",  # Phenol (toxic)
        "CC(C)CCO",  # Isopentanol
        "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Anthracene (toxic)
    ]
    
    # Labels (1 = toxic, 0 = non-toxic)
    labels = [0, 0, 0, 1, 0, 1, 0, 1]
    
    # Duplicate data to make it larger
    train_smiles = sample_molecules * 100
    train_labels = np.array(labels * 100)
    
    test_smiles = sample_molecules * 20
    test_labels = np.array(labels * 20)
    
    print(f"Created {len(train_smiles)} training samples, {len(test_smiles)} test samples")
    
    return train_smiles, train_labels, test_smiles, test_labels

# ============================================================================
# STEP 3: Feature Engineering
# ============================================================================

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES string to Morgan fingerprint (circular fingerprint)
    This is a common molecular representation in cheminformatics
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    
    # Morgan fingerprint (similar to ECFP - Extended Connectivity Fingerprint)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def calculate_molecular_descriptors(smiles):
    """
    Calculate physicochemical properties of molecules
    These are interpretable features that chemists understand
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(10)
    
    descriptors = [
        Descriptors.MolWt(mol),              # Molecular weight
        Descriptors.MolLogP(mol),            # Lipophilicity (log P)
        Descriptors.NumHDonors(mol),         # Hydrogen bond donors
        Descriptors.NumHAcceptors(mol),      # Hydrogen bond acceptors
        Descriptors.TPSA(mol),               # Topological polar surface area
        Descriptors.NumRotatableBonds(mol),  # Rotatable bonds
        Descriptors.NumAromaticRings(mol),   # Aromatic rings
        Descriptors.NumAliphaticRings(mol),  # Aliphatic rings
        Descriptors.FractionCsp3(mol),       # Fraction of sp3 carbons
        Descriptors.NumHeteroatoms(mol),     # Number of heteroatoms
    ]
    
    return np.array(descriptors)

def featurize_molecules(smiles_list, use_descriptors=True):
    """
    Convert list of SMILES to feature matrix
    """
    print(f"Featurizing {len(smiles_list)} molecules...")
    
    fingerprints = []
    descriptors = []
    
    for smiles in smiles_list:
        fp = smiles_to_fingerprint(smiles)
        fingerprints.append(fp)
        
        if use_descriptors:
            desc = calculate_molecular_descriptors(smiles)
            descriptors.append(desc)
    
    fingerprints = np.array(fingerprints)
    
    if use_descriptors:
        descriptors = np.array(descriptors)
        # Combine fingerprints and descriptors
        features = np.hstack([fingerprints, descriptors])
        print(f"Feature shape: {features.shape} (fingerprints + descriptors)")
    else:
        features = fingerprints
        print(f"Feature shape: {features.shape} (fingerprints only)")
    
    return features


# STEP 4: Model Training


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest classifier
    """
    print("\n=== Training Random Forest ===")
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return rf_model, auc

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost classifier
    """
    print("\n=== Training XGBoost ===")
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Evaluation
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return xgb_model, auc


# STEP 5: Feature Importance Analysis


def analyze_feature_importance(model, feature_names=None, top_n=10):
    """
    Analyze which molecular features are most important for predictions
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Get top N important features
        indices = np.argsort(importances)[-top_n:][::-1]
        
        print(f"\nTop {top_n} Most Important Features:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), importances[indices])
        plt.yticks(range(top_n), [feature_names[idx] for idx in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top Molecular Features for Toxicity Prediction')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance plot saved as 'feature_importance.png'")


# MAIN EXECUTION


def main():
    print("=" * 70)
    print("MOLECULAR TOXICITY PREDICTION PROJECT")
    print("=" * 70)
    
    # Load data
    train_smiles, train_labels, test_smiles, test_labels = load_tox21_data()
    
    # Featurize molecules
    X_train = featurize_molecules(train_smiles, use_descriptors=True)
    X_test = featurize_molecules(test_smiles, use_descriptors=True)
    
    print(f"\nClass distribution in training set:")
    print(f"Non-toxic (0): {np.sum(train_labels == 0)} ({np.mean(train_labels == 0)*100:.1f}%)")
    print(f"Toxic (1): {np.sum(train_labels == 1)} ({np.mean(train_labels == 1)*100:.1f}%)")
    
    # Train models
    rf_model, rf_auc = train_random_forest(X_train, train_labels, X_test, test_labels)
    xgb_model, xgb_auc = train_xgboost(X_train, train_labels, X_test, test_labels)
    
    # Feature importance analysis
    descriptor_names = [
        'MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 
        'RotBonds', 'ArRings', 'AlRings', 'Fsp3', 'Heteroatoms'
    ]
    
    # Use the better performing model
    best_model = rf_model if rf_auc > xgb_auc else xgb_model
    model_name = "Random Forest" if rf_auc > xgb_auc else "XGBoost"
    
    print(f"\n{model_name} performed better (AUC: {max(rf_auc, xgb_auc):.4f})")
    
    # Analyze descriptor importance (last 10 features)
    analyze_feature_importance(best_model, 
                               feature_names=[f"FP_{i}" for i in range(2048)] + descriptor_names,
                               top_n=15)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {model_name}")
    print(f"Best ROC-AUC: {max(rf_auc, xgb_auc):.4f}")
    print("\nYou can now use this project for your interview response!")
    print("=" * 70)

if __name__ == "__main__":
    main()