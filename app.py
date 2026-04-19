# ============================================================================
# CREDIT RISK EVALUATION SYSTEM - FULLY OPTIMIZED
# LENDING CLUB DATASET 2007-2018
# 10 MACHINE LEARNING MODELS WITH PLOT CACHING
# ============================================================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# === IMPORT SHAP ===
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP available")
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from collections import Counter

# === MODELS ===
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier

# === ADVANCED ENSEMBLE METHODS ===
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    print("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
    print("LightGBM available")
except ImportError:
    print("LightGBM not available")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
    print("CatBoost available")
except ImportError:
    print("CatBoost not available")

import joblib

app = Flask(__name__)

# ============================================================================
# GLOBAL CACHED VARIABLES
# ============================================================================

models_dict = {}
model_metrics = {}
confusion_matrices_dict = {}
model_cv_scores = {}
scaler = None
label_encoders = {}
feature_names = []
X_train, X_test, y_train, y_test = None, None, None, None
df_raw = None
shap_explainer = None
training_complete = False
data_loaded = False

# ========== PLOT CACHING (CRITICAL FOR PERFORMANCE) ==========
cached_plots = {
    "distribution": None,
    "correlation": None,
    "roc_curve": None,
    "confusion_matrices": {},
    "feature_importance": None,
    "feature_importance_data": None,
    "shap_summary": None,
    "shap_bar": None,
    "shap_force": None,
    "comparison_chart": None
}

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_lending_club_dataset():
    """Load the Lending Club dataset - CACHED"""
    global df_raw, data_loaded
    
    if data_loaded and df_raw is not None:
        print("Using cached dataset")
        return df_raw
    
    print("="*60)
    print("LOADING LENDING CLUB DATASET")
    print("="*60)
    
    possible_paths = [
        'accepted_2007_to_2018Q4.csv',
        'accepted_2007_to_2018Q4.csv.gz',
        'data/accepted_2007_to_2018Q4.csv',
        'data/accepted_2007_to_2018Q4.csv.gz'
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print(f"Dataset not found! Using synthetic data.")
        df_raw = create_synthetic_dataset()
        data_loaded = True
        return df_raw
    
    print(f"Found dataset at: {dataset_path}")
    file_size = os.path.getsize(dataset_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    try:
        columns_to_load = [
            'loan_amnt', 'term', 'int_rate', 'installment', 'grade',
            'emp_length', 'home_ownership', 'annual_inc', 'loan_status',
            'purpose', 'dti', 'delinq_2yrs', 'inq_last_6mths', 'open_acc',
            'pub_rec', 'revol_util', 'total_acc', 'collections_12_mths_ex_med', 'mort_acc'
        ]
        
        print("Loading dataset...")
        
        if dataset_path.endswith('.gz'):
            df = pd.read_csv(dataset_path, compression='gzip', 
                           usecols=columns_to_load, low_memory=False, nrows=100000)
        else:
            df = pd.read_csv(dataset_path, usecols=columns_to_load, 
                           low_memory=False, nrows=100000)
        
        print(f"Dataset loaded! Shape: {df.shape}")
        df_raw = clean_lending_club_data(df)
        data_loaded = True
        return df_raw
        
    except Exception as e:
        print(f"   Error: {e}")
        df_raw = create_synthetic_dataset()
        data_loaded = True
        return df_raw

def clean_lending_club_data(df):
    """Clean and prepare Lending Club data."""
    print("\nCLEANING DATASET")
    
    df_clean = df.copy()
    df_clean = df_clean[df_clean['loan_status'].notna()]
    
    default_statuses = [
        'Charged Off', 'Default', 'Late (31-120 days)',
        'In Grace Period', 'Does not meet the credit policy. Status:Charged Off'
    ]
    
    df_clean['default'] = df_clean['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)
    df_clean['default_label'] = df_clean['default'].map({1: 'Default', 0: 'Fully Paid'})
    
    print(f"   Default rate: {df_clean['default'].mean():.2%}")
    print(f"   Total loans: {len(df_clean):,}")
    print(f"   Defaulted loans: {df_clean['default'].sum():,}")
    print(f"   Non-default loans: {(df_clean['default'] == 0).sum():,}")
    
    if 'int_rate' in df_clean.columns:
        df_clean['interest_rate'] = df_clean['int_rate'].astype(str).str.replace('%', '').astype(float)
    
    if 'revol_util' in df_clean.columns:
        df_clean['revolving_utilization'] = df_clean['revol_util'].astype(str).str.replace('%', '').astype(float)
    
    if 'emp_length' in df_clean.columns:
        emp_length_map = {
            '< 1 year': '< 1 year', '1 year': '1 year', '2 years': '2 years',
            '3 years': '3 years', '4 years': '4 years', '5 years': '5 years',
            '6 years': '6 years', '7 years': '7 years', '8 years': '8 years',
            '9 years': '9 years', '10+ years': '10+ years', 'n/a': '< 1 year'
        }
        df_clean['employment_length'] = df_clean['emp_length'].map(emp_length_map)
    
    if 'grade' in df_clean.columns:
        grade_scores = {'A': 750, 'B': 700, 'C': 650, 'D': 600, 'E': 550, 'F': 500, 'G': 450}
        df_clean['credit_score'] = df_clean['grade'].map(grade_scores)
        if 'interest_rate' in df_clean.columns:
            df_clean['credit_score'] = df_clean['credit_score'] - (df_clean['interest_rate'] - 12) * 5
            df_clean['credit_score'] = df_clean['credit_score'].clip(300, 850)
    
    column_mapping = {
        'loan_amnt': 'loan_amount', 'annual_inc': 'annual_income',
        'dti': 'debt_to_income', 'delinq_2yrs': 'delinquencies',
        'open_acc': 'num_credit_lines', 'total_acc': 'total_credit_lines',
        'purpose': 'loan_purpose', 'home_ownership': 'home_ownership',
        'pub_rec': 'public_records', 'inq_last_6mths': 'credit_inquiries',
        'mort_acc': 'mortgage_accounts', 'collections_12_mths_ex_med': 'collections'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df_clean.columns:
            df_clean[new_name] = df_clean[old_name]
    
    final_columns = [
        'loan_amount', 'interest_rate', 'annual_income', 'employment_length',
        'home_ownership', 'loan_purpose', 'debt_to_income', 'delinquencies',
        'num_credit_lines', 'revolving_utilization', 'total_credit_lines',
        'credit_score', 'public_records', 'credit_inquiries', 
        'mortgage_accounts', 'collections', 'default', 'default_label'
    ]
    
    existing_columns = [col for col in final_columns if col in df_clean.columns]
    df_final = df_clean[existing_columns].copy()
    
    print(f"\n   Original shape: {df_final.shape}")
    
    numerical_cols = ['loan_amount', 'interest_rate', 'annual_income', 'debt_to_income', 
                      'delinquencies', 'num_credit_lines', 'revolving_utilization', 
                      'total_credit_lines', 'credit_score', 'public_records',
                      'credit_inquiries', 'mortgage_accounts', 'collections']
    
    for col in numerical_cols:
        if col in df_final.columns and df_final[col].isnull().any():
            missing_count = df_final[col].isnull().sum()
            median_val = df_final[col].median()
            df_final[col].fillna(median_val, inplace=True)
            print(f"      Filled {missing_count} missing in {col} with median: {median_val:.2f}")
    
    categorical_cols = ['employment_length', 'home_ownership', 'loan_purpose']
    for col in categorical_cols:
        if col in df_final.columns and df_final[col].isnull().any():
            missing_count = df_final[col].isnull().sum()
            mode_val = df_final[col].mode()[0] if not df_final[col].mode().empty else 'Unknown'
            df_final[col].fillna(mode_val, inplace=True)
            print(f"      Filled {missing_count} missing in {col} with mode: {mode_val}")
    
    if df_final.isnull().sum().sum() > 0:
        print(f"   WARNING: Still have {df_final.isnull().sum().sum()} NaN values. Filling with 0...")
        df_final = df_final.fillna(0)
    
    df_final = df_final.dropna(subset=['default'])
    
    print(f"\nData cleaning complete! Final shape: {df_final.shape}")
    return df_final

def create_synthetic_dataset():
    """Create synthetic dataset as fallback."""
    print("Creating synthetic dataset...")
    np.random.seed(42)
    n = 15000
    
    df = pd.DataFrame({
        'loan_amount': np.random.randint(1000, 40000, n),
        'interest_rate': np.random.uniform(5, 24, n).round(2),
        'annual_income': np.random.randint(20000, 200000, n),
        'employment_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', 
                                                '4 years', '5 years', '6 years', '7 years',
                                                '8 years', '9 years', '10+ years'], n),
        'home_ownership': np.random.choice(['RENT', 'MORTGAGE', 'OWN', 'OTHER'], n),
        'loan_purpose': np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement',
                                          'medical', 'business', 'other'], n),
        'debt_to_income': np.random.uniform(0, 40, n).round(1),
        'delinquencies': np.random.choice([0, 1, 2], n, p=[0.85, 0.1, 0.05]),
        'num_credit_lines': np.random.randint(1, 20, n),
        'revolving_utilization': np.random.uniform(0, 100, n).round(1),
        'total_credit_lines': np.random.randint(1, 40, n),
        'credit_score': np.random.randint(300, 850, n),
        'public_records': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        'credit_inquiries': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.1, 0.1]),
        'mortgage_accounts': np.random.choice([0, 1, 2], n, p=[0.7, 0.25, 0.05]),
        'collections': np.random.choice([0, 1], n, p=[0.92, 0.08])
    })
    
    risk_score = (
        (df['credit_score'] < 600) * 0.30 +
        (df['debt_to_income'] > 35) * 0.20 +
        (df['interest_rate'] > 18) * 0.15 +
        (df['delinquencies'] > 0) * 0.10 +
        (df['credit_inquiries'] > 2) * 0.10 +
        (df['public_records'] > 0) * 0.10 +
        (df['collections'] > 0) * 0.05
    )
    
    default_prob = 1 / (1 + np.exp(-(risk_score - 0.5) * 5))
    df['default'] = np.random.binomial(1, default_prob)
    df['default_label'] = df['default'].map({1: 'Default', 0: 'Fully Paid'})
    
    print(f"Synthetic dataset created! Shape: {df.shape}")
    return df

# ============================================================================
# FEATURE SELECTION
# ============================================================================

def perform_feature_selection(X, y, n_features=16):
    """Perform feature selection using 4 techniques."""
    print("\n" + "="*60)
    print("FEATURE SELECTION - Using All 4 Techniques")
    print("="*60)
    
    f_selector = SelectKBest(score_func=f_classif, k=n_features)
    f_selector.fit(X, y)
    f_scores = pd.DataFrame({'feature': X.columns, 'f_score': f_selector.scores_}).sort_values('f_score', ascending=False)
    print(f"   Top features: {f_scores.head(5)['feature'].tolist()}")
    
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    mi_selector.fit(X, y)
    mi_scores = pd.DataFrame({'feature': X.columns, 'mi_score': mi_selector.scores_}).sort_values('mi_score', ascending=False)
    print(f"   Top features: {mi_scores.head(5)['feature'].tolist()}")
    
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    rf_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_temp.feature_importances_}).sort_values('importance', ascending=False)
    print(f"   Top features: {rf_importance.head(5)['feature'].tolist()}")
    
    X_sample = X.sample(min(5000, len(X)), random_state=42)
    y_sample = y.loc[X_sample.index]
    lr_base = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator=lr_base, n_features_to_select=n_features)
    rfe.fit(X_sample, y_sample)
    rfe_features = X.columns[rfe.support_].tolist()
    print(f"   RFE selected: {rfe_features}")
    
    all_features = []
    all_features.extend(f_scores.head(10)['feature'].tolist())
    all_features.extend(mi_scores.head(10)['feature'].tolist())
    all_features.extend(rf_importance.head(10)['feature'].tolist())
    all_features.extend(rfe_features)
    
    feature_counts = Counter(all_features)
    final_features = [feat for feat, count in feature_counts.most_common(n_features)]
    
    print(f"\nSelected {len(final_features)} features: {final_features}")
    return final_features

# ============================================================================
# MODEL DEFINITION
# ============================================================================

def get_all_models():
    """Define all machine learning models."""
    models = {}
    
    models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    models['Decision Tree'] = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss', scale_pos_weight=4.48)
    
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1, class_weight='balanced')
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(n_estimators=100, learning_rate=0.1, depth=5, random_state=42, verbose=False, auto_class_weights='Balanced')
    
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=100, learning_rate=1.0, random_state=42)
    models['Extra Trees'] = ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    models['Neural Network'] = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True)
    
    return models

# ============================================================================
# SHAP FUNCTIONS
# ============================================================================

def create_shap_explainer(model, X_sample):
    """Create SHAP explainer for the model."""
    global shap_explainer
    
    if not SHAP_AVAILABLE:
        return None
    
    try:
        if isinstance(model, (RandomForestClassifier, DecisionTreeClassifier, 
                              XGBClassifier, LGBMClassifier, CatBoostClassifier,
                              GradientBoostingClassifier, ExtraTreesClassifier)):
            shap_explainer = shap.TreeExplainer(model)
            print("SHAP TreeExplainer created")
        else:
            shap_explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
            print("SHAP KernelExplainer created")
        return True
    except Exception as e:
        print(f"SHAP explainer creation failed: {e}")
        return False

# ============================================================================
# PLOT GENERATION FUNCTIONS (WITH CACHING)
# ============================================================================

def generate_distribution_plot():
    """Generate distribution plot - CACHED"""
    global cached_plots, df_raw
    
    if cached_plots["distribution"] is not None:
        return cached_plots["distribution"]
    
    print("Generating distribution plot (first time only)...")
    
    plt.figure(figsize=(10, 6))
    counts = df_raw['default_label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = plt.bar(counts.index, counts.values, color=colors)
    plt.title('Loan Default Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    for bar, value in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(value), ha='center', fontweight='bold')
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    cached_plots["distribution"] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return cached_plots["distribution"]

def generate_correlation_plot():
    """Generate correlation plot - CACHED"""
    global cached_plots, df_raw
    
    if cached_plots["correlation"] is not None:
        return cached_plots["correlation"]
    
    print("Generating correlation plot (first time only)...")
    
    numerical_cols = df_raw.select_dtypes(include=[np.number]).columns
    numerical_cols = [col for col in numerical_cols if col != 'default']
    
    if len(numerical_cols) > 1:
        plt.figure(figsize=(14, 12))
        corr_matrix = df_raw[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, annot_kws={'size': 7})
        plt.title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        cached_plots["correlation"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    else:
        cached_plots["correlation"] = None
    
    return cached_plots["correlation"]

def generate_roc_curve():
    """Generate ROC curve - CACHED"""
    global cached_plots, models_dict, X_test, y_test, model_metrics
    
    if cached_plots["roc_curve"] is not None:
        return cached_plots["roc_curve"]
    
    print("Generating ROC curve (first time only)...")
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(models_dict)))
    
    for (name, model), color in zip(models_dict.items(), colors):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = model_metrics[name]['ROC-AUC']
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=2, color=color)
    
    plt.plot([0,1], [0,1], 'k--', label='Random (AUC=0.5)', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    cached_plots["roc_curve"] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return cached_plots["roc_curve"]

def generate_confusion_matrix(name, cm):
    """Generate confusion matrix - CACHED per model"""
    global cached_plots
    
    if name in cached_plots["confusion_matrices"]:
        return cached_plots["confusion_matrices"][name]
    
    print(f"Generating confusion matrix for {name} (first time only)...")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Non-Default', 'Default'],
               yticklabels=['Non-Default', 'Default'])
    plt.title(f'Confusion Matrix - {name}', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
    img.seek(0)
    cached_plots["confusion_matrices"][name] = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return cached_plots["confusion_matrices"][name]

def generate_feature_importance():
    """Generate feature importance plot - CACHED"""
    global cached_plots, models_dict, feature_names
    
    if cached_plots["feature_importance"] is not None:
        return cached_plots["feature_importance"], cached_plots["feature_importance_data"]
    
    print("Generating feature importance (first time only)...")
    
    if 'Random Forest' in models_dict:
        rf_model = models_dict['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        cached_plots["feature_importance_data"] = feature_importance.to_dict('records')
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top Features - Random Forest', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        cached_plots["feature_importance"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    else:
        cached_plots["feature_importance"] = None
        cached_plots["feature_importance_data"] = None
    
    return cached_plots["feature_importance"], cached_plots["feature_importance_data"]

def generate_shap_plots():
    """Generate SHAP plots - CACHED (runs once, very heavy)"""
    global cached_plots, shap_explainer, X_test, feature_names
    
    if cached_plots["shap_summary"] is not None:
        return cached_plots["shap_summary"], cached_plots["shap_bar"], cached_plots["shap_force"]
    
    if not SHAP_AVAILABLE or shap_explainer is None or X_test is None:
        return None, None, None
    
    print("Generating SHAP plots (first time only - this may take 5-10 seconds)...")
    
    try:
        # Use only 50 samples for SHAP (faster)
        X_sample = X_test[:50]
        shap_values = shap_explainer.shap_values(X_sample)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        cached_plots["shap_summary"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Bar Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=150, bbox_inches='tight')
        img.seek(0)
        cached_plots["shap_bar"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        # Force Plot
        print("   Generating SHAP Force Plot (making it wider for better visibility)...")
        
        # Pick a single prediction for force plot
        single_shap = shap_values[0]
        single_sample = X_sample.iloc[0:1]
        expected_value = shap_explainer.expected_value[1] if isinstance(shap_explainer.expected_value, list) else shap_explainer.expected_value
        
        # Create force plot with custom figure size (wider)
        plt.figure(figsize=(20, 4))
        
        shap.force_plot(
            expected_value, 
            single_shap, 
            single_sample, 
            feature_names=feature_names, 
            matplotlib=True, 
            show=False,
            text_rotation=0
        )
        
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=200, bbox_inches='tight')
        img.seek(0)
        cached_plots["shap_force"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        print("SHAP plots cached successfully!")
        
    except Exception as e:
        print(f"SHAP plots failed: {e}")
        cached_plots["shap_summary"] = None
        cached_plots["shap_bar"] = None
        cached_plots["shap_force"] = None
    
    return cached_plots["shap_summary"], cached_plots["shap_bar"], cached_plots["shap_force"]

def generate_comparison_chart():
    """Generate model comparison chart - CACHED"""
    global cached_plots, model_metrics
    
    if cached_plots["comparison_chart"] is not None:
        return cached_plots["comparison_chart"]
    
    print("Generating comparison chart (first time only)...")
    
    try:
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            if idx < 5:
                metric_values = {name: metrics[metric] for name, metrics in model_metrics.items()}
                sorted_models = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
                
                names = [m[0] for m in sorted_models]
                values = [m[1] for m in sorted_models]
                colors_bar = ['#e74c3c' if v == max(values) else '#3498db' for v in values]
                
                axes[idx].barh(names, values, color=colors_bar)
                axes[idx].set_xlabel(metric)
                axes[idx].set_title(f'Model Comparison - {metric}', fontweight='bold')
                axes[idx].invert_yaxis()
        
        axes[5].remove()
        plt.tight_layout()
        img = io.BytesIO()
        plt.savefig(img, format='png', dpi=100, bbox_inches='tight')
        img.seek(0)
        cached_plots["comparison_chart"] = base64.b64encode(img.getvalue()).decode()
        plt.close()
    except Exception as e:
        cached_plots["comparison_chart"] = None
    
    return cached_plots["comparison_chart"]

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_all_models():
    """Train all models - RUNS ONCE"""
    global models_dict, model_metrics, confusion_matrices_dict, model_cv_scores
    global scaler, label_encoders, feature_names, selected_features
    global X_train, X_test, y_train, y_test, df_raw, shap_explainer, training_complete
    
    print("\n" + "="*60)
    print("TRAINING MACHINE LEARNING MODELS")
    print("="*60)
    
    df_raw = load_lending_club_dataset()
    
    if len(df_raw) < 100:
        print("ERROR: Not enough data!")
        return
    
    y = df_raw['default']
    X = df_raw.drop(['default', 'default_label'], axis=1)
    
    print(f"\nTotal features: {len(X.columns)}")
    print(f"   Features: {list(X.columns)}")
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"   Encoded {col}: {len(le.classes_)} categories")
    
    print(f"\nBEFORE feature selection: {X.shape[1]} features")
    selected_features = perform_feature_selection(X, y, n_features=16)
    X = X[selected_features]
    feature_names = selected_features
    print(f"AFTER feature selection: {X.shape[1]} features")
    print(f"Selected features: {feature_names}")
    
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.isnull().sum().sum() > 0:
        print("\n   Filling remaining NaNs with column means...")
        X = X.fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nDATASET SPLIT:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Default rate (training): {y_train.mean():.2%}")
    print(f"   Default rate (test): {y_test.mean():.2%}")
    
    all_models = get_all_models()
    
    print("\nTRAINING MODELS...")
    print("-" * 60)
    
    for name, model in all_models.items():
        try:
            print(f"\n   Training {name}...")
            model.fit(X_train, y_train)
            models_dict[name] = model
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            model_metrics[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_proba)
            }
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
            model_cv_scores[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
            
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices_dict[name] = cm
            
            print(f"      Accuracy: {model_metrics[name]['Accuracy']:.4f}")
            print(f"      ROC-AUC: {model_metrics[name]['ROC-AUC']:.4f}")
            print(f"      CV ROC-AUC: {cv_scores.mean():.4f} (+/-{cv_scores.std():.4f})")
            
        except Exception as e:
            print(f"      Failed: {str(e)[:100]}")
            continue
    
    print("\n" + "="*60)
    print(f"Successfully trained {len(models_dict)} out of {len(all_models)} models")
    print("="*60)
    
    if models_dict:
        best = max(model_metrics, key=lambda x: model_metrics[x]['ROC-AUC'])
        print(f"\nBEST MODEL: {best}")
        print(f"   ROC-AUC: {model_metrics[best]['ROC-AUC']:.4f}")
        print(f"   Accuracy: {model_metrics[best]['Accuracy']:.4f}")
        
        if SHAP_AVAILABLE:
            best_model = models_dict[best]
            create_shap_explainer(best_model, X_test[:100])
    
    training_complete = True

# ============================================================================
# FLASK ROUTES (USING CACHED PLOTS)
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    global df_raw, feature_names, training_complete
    
    if not training_complete and not models_dict:
        train_all_models()
    
    if df_raw is None:
        return "<h1>Loading...</h1><p>Please wait. Refresh in a moment.</p>"
    
    total_samples = len(df_raw)
    default_rate = df_raw['default'].mean() * 100
    features_count = len(feature_names) if feature_names else len(df_raw.columns) - 2
    
    images = {
        "distribution": generate_distribution_plot(),
        "correlation": generate_correlation_plot()
    }
    
    df_display = df_raw.head(10).copy()
    for col in df_display.select_dtypes(include=['float64']).columns:
        df_display[col] = df_display[col].round(2)
    df_head_html = df_display.to_html(classes='table table-striped table-hover', index=False)
    
    return render_template('dashboard.html',
                         total_samples=total_samples,
                         default_rate=default_rate,
                         features_count=features_count,
                         images=images,
                         df_head=df_head_html)

@app.route('/model-results')
def model_results():
    global model_metrics, confusion_matrices_dict, X_test, y_test, models_dict, training_complete
    
    if not training_complete and not models_dict:
        train_all_models()
    
    if not model_metrics:
        return "<h1>Loading...</h1><p>Models are still loading. Please refresh.</p>"
    
    # Calculate best ROC-AUC to highlight the best model
    best_auc = max(metrics['ROC-AUC'] for metrics in model_metrics.values())
    
    # Generate all cached plots
    cm_images = {}
    for name, cm in confusion_matrices_dict.items():
        cm_images[name] = generate_confusion_matrix(name, cm)
    
    roc_curve_img = generate_roc_curve()
    feature_importance_img, feature_importance_data = generate_feature_importance()
    comparison_chart = generate_comparison_chart()
    shap_summary, shap_bar, shap_force = generate_shap_plots()
    
    return render_template('model_results.html',
                         results=model_metrics,
                         cm_images=cm_images,
                         roc_curve_img=roc_curve_img,
                         feature_importance_img=feature_importance_img,
                         feature_importance_data=feature_importance_data,
                         comparison_chart=comparison_chart,
                         shap_summary_plot=shap_summary,
                         shap_bar_plot=shap_bar,
                         shap_force_plot=shap_force,
                         num_models=len(models_dict),
                         best_auc=best_auc)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global models_dict, scaler, label_encoders, feature_names, model_metrics, shap_explainer, training_complete
    
    if not training_complete and not models_dict:
        train_all_models()
    
    if not models_dict:
        return "<h1>Loading...</h1><p>Models are still loading. Please refresh.</p>"
    
    best_model_name = max(model_metrics, key=lambda x: model_metrics[x]['ROC-AUC'])
    prediction_model = models_dict.get(best_model_name, list(models_dict.values())[0])
    
    shap_contributions = None
    
    if request.method == 'POST':
        try:
            print("\n" + "="*60)
            print("PREDICTION REQUEST")
            print("="*60)
            
            loan_amount = float(request.form.get('loan_amount', 0))
            interest_rate = float(request.form.get('interest_rate', 0))
            annual_income = float(request.form.get('annual_income', 0))
            employment_length = request.form.get('employment_length', '5 years')
            credit_score = float(request.form.get('credit_score', 0))
            debt_to_income = float(request.form.get('debt_to_income', 0))
            num_credit_lines = float(request.form.get('num_credit_lines', 0))
            delinquencies = float(request.form.get('delinquencies', 0))
            revolving_utilization = float(request.form.get('revolving_utilization', 0))
            total_credit_lines = float(request.form.get('total_credit_lines', 0))
            loan_purpose = request.form.get('loan_purpose', 'debt_consolidation')
            home_ownership = request.form.get('home_ownership', 'RENT')
            credit_inquiries = float(request.form.get('credit_inquiries', 1))
            mortgage_accounts = float(request.form.get('mortgage_accounts', 0))
            public_records = float(request.form.get('public_records', 0))
            collections = float(request.form.get('collections', 0))
            
            print(f"Applicant: ${loan_amount:,.0f} loan, {interest_rate}% rate, {credit_score} credit score")
            
            values_list = []
            for feature in feature_names:
                if feature == 'loan_amount':
                    values_list.append(loan_amount)
                elif feature == 'interest_rate':
                    values_list.append(interest_rate)
                elif feature == 'annual_income':
                    values_list.append(annual_income)
                elif feature == 'employment_length':
                    if feature in label_encoders:
                        val = employment_length if employment_length in label_encoders[feature].classes_ else label_encoders[feature].classes_[0]
                        values_list.append(label_encoders[feature].transform([val])[0])
                    else:
                        values_list.append(5)
                elif feature == 'home_ownership':
                    if home_ownership in label_encoders[feature].classes_:
                        values_list.append(label_encoders[feature].transform([home_ownership])[0])
                    else:
                        values_list.append(label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0])
                elif feature == 'loan_purpose':
                    if loan_purpose in label_encoders[feature].classes_:
                        values_list.append(label_encoders[feature].transform([loan_purpose])[0])
                    else:
                        values_list.append(label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0])
                elif feature == 'debt_to_income':
                    values_list.append(debt_to_income)
                elif feature == 'delinquencies':
                    values_list.append(delinquencies)
                elif feature == 'num_credit_lines':
                    values_list.append(num_credit_lines)
                elif feature == 'revolving_utilization':
                    values_list.append(revolving_utilization)
                elif feature == 'total_credit_lines':
                    values_list.append(total_credit_lines)
                elif feature == 'credit_score':
                    values_list.append(credit_score)
                elif feature == 'credit_inquiries':
                    values_list.append(credit_inquiries)
                elif feature == 'mortgage_accounts':
                    values_list.append(mortgage_accounts)
                elif feature == 'public_records':
                    values_list.append(public_records)
                elif feature == 'collections':
                    values_list.append(collections)
                else:
                    values_list.append(0)
            
            input_df_raw = pd.DataFrame([values_list], columns=feature_names)
            input_scaled = scaler.transform(input_df_raw)
            input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
            
            probability = prediction_model.predict_proba(input_scaled_df)[0][1]
            risk_score = probability * 100
            
            print(f"Prediction: Default Probability = {probability:.2%}, Risk Score = {risk_score:.1f}%")
            
            if SHAP_AVAILABLE and shap_explainer is not None:
                try:
                    input_array = input_scaled_df.values
                    shap_values = shap_explainer.shap_values(input_array)
                    
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                        shap_values = shap_values[0]
                    
                    shap_contributions = []
                    raw_values = input_df_raw.iloc[0].to_dict()
                    
                    for i, feature in enumerate(feature_names):
                        contribution = float(shap_values[i]) if i < len(shap_values) else 0.0
                        feature_value = raw_values.get(feature, 0)
                        
                        shap_contributions.append({
                            'feature': feature,
                            'value': feature_value,
                            'contribution': contribution,
                            'abs_contribution': abs(contribution),
                            'direction': 'positive' if contribution > 0 else 'negative'
                        })
                    
                    shap_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
                    print(f"SHAP explanation generated")
                    
                except Exception as e:
                    print(f"SHAP explanation failed: {e}")
            
            if risk_score < 15:
                risk_level = "Low Risk"
                recommendation = "Application Approved - Proceed with standard loan approval"
                color = "success"
            elif risk_score < 30:
                risk_level = "Low-Medium Risk"
                recommendation = "Application Approved - Consider standard rate"
                color = "info"
            elif risk_score < 50:
                risk_level = "Medium Risk"
                recommendation = "Application Under Review - Consider higher interest rate"
                color = "warning"
            elif risk_score < 70:
                risk_level = "High Risk"
                recommendation = "Application Requires Additional Verification - High default probability"
                color = "danger"
            else:
                risk_level = "Very High Risk"
                recommendation = "Application Rejected - Very high probability of default"
                color = "dark"
            
            all_predictions = {}
            for name, model in models_dict.items():
                try:
                    all_predictions[name] = model.predict_proba(input_scaled_df)[0][1]
                except:
                    all_predictions[name] = None
            
            applicant_data = {
                'loan_amount': loan_amount,
                'interest_rate': interest_rate,
                'annual_income': annual_income,
                'employment_length': employment_length,
                'credit_score': credit_score,
                'debt_to_income': debt_to_income,
                'num_credit_lines': num_credit_lines,
                'delinquencies': delinquencies,
                'revolving_utilization': revolving_utilization,
                'total_credit_lines': total_credit_lines,
                'loan_purpose': loan_purpose,
                'home_ownership': home_ownership,
                'credit_inquiries': credit_inquiries,
                'mortgage_accounts': mortgage_accounts,
                'public_records': public_records,
                'collections': collections
            }
            
            return render_template('predict.html',
                                 prediction=int(probability > 0.5),
                                 probability=probability,
                                 risk_score=risk_score,
                                 risk_level=risk_level,
                                 recommendation=recommendation,
                                 color=color,
                                 applicant_data=applicant_data,
                                 best_model=best_model_name,
                                 all_predictions=all_predictions,
                                 shap_contributions=shap_contributions,
                                 shap_available=SHAP_AVAILABLE)
        
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return render_template('predict.html', error=f"Prediction error: {str(e)}")
    
    return render_template('predict.html', shap_available=SHAP_AVAILABLE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    global models_dict, scaler, label_encoders, feature_names, model_metrics, training_complete
    
    if not training_complete and not models_dict:
        train_all_models()
    
    if not models_dict:
        return jsonify({'error': 'Models not ready yet'}), 503
    
    best_model_name = max(model_metrics, key=lambda x: model_metrics[x]['ROC-AUC'])
    prediction_model = models_dict.get(best_model_name, list(models_dict.values())[0])
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        values_list = []
        for feature in feature_names:
            if feature in data:
                if feature in label_encoders:
                    val = data[feature]
                    if val in label_encoders[feature].classes_:
                        values_list.append(label_encoders[feature].transform([val])[0])
                    else:
                        values_list.append(label_encoders[feature].transform([label_encoders[feature].classes_[0]])[0])
                else:
                    values_list.append(float(data[feature]))
            else:
                values_list.append(0)
        
        input_df = pd.DataFrame([values_list], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        
        probability = prediction_model.predict_proba(input_scaled_df)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        return jsonify({
            'prediction': prediction,
            'probability': float(probability),
            'risk_score': float(probability * 100),
            'risk_level': 'High Risk' if probability > 0.5 else 'Low Risk',
            'model_used': best_model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("CREDIT RISK EVALUATION SYSTEM - FULLY OPTIMIZED")
    print("10 MACHINE LEARNING MODELS WITH PLOT CACHING")
    print("="*60)
    print("\nALL 16 FEATURES + 2 TARGETS = 18 TOTAL COLUMNS")
    print("\nFEATURE SELECTION: 4 techniques (ANOVA, MI, RF, RFE)")
    print("CLASS IMBALANCE HANDLING: Active")
    print("SHAP EXPLAINABLE AI: Enabled")
    print("PLOT CACHING: Enabled (first load slow, subsequent loads instant)")
    print("\nFirst run: 10-15 minutes (training)")
    print("First page load: 3-6 seconds (generating plots)")
    print("Subsequent page loads: 1-2 seconds (cached plots)")
    print("="*60)
    
    train_all_models()
    
    print("\nServer is running!")
    print("Web Interface: http://127.0.0.1:5000")
    print("API Endpoint: http://127.0.0.1:5000/api/predict")
    print("="*60)
    
    if __name__ == "__main__":
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)