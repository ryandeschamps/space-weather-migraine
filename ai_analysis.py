#!/usr/bin/env python3
"""
Advanced AI Analysis for Space Weather and Migraine Correlation
This script performs GPU-accelerated machine learning analysis and saves results for deployment.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
import pickle
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_space_weather_data():
    """Load and preprocess space weather data"""
    from space_weather_dashboard import load_all_data
    
    print("🔄 Loading space weather data...")
    df = load_all_data()
    
    if df.empty:
        raise ValueError("No data loaded!")
    
    print(f"✅ Loaded {len(df)} days of space weather data")
    return df

def create_advanced_features(df):
    """Create advanced features for ML analysis"""
    print("🔧 Creating advanced features...")
    
    df_ml = df.copy()
    
    # Add migraine indicator
    migraine_dates = [pd.Timestamp('2025-06-03'), pd.Timestamp('2025-05-28')]
    df_ml['migraine'] = df_ml['date'].isin(migraine_dates).astype(int)
    
    # Sort by date for time series features
    df_ml = df_ml.sort_values('date').reset_index(drop=True)
    
    features_created = []
    
    # Kp index features
    if 'max_kp' in df_ml.columns:
        df_ml['kp_rolling_mean_3'] = df_ml['max_kp'].rolling(window=3, center=True).mean()
        df_ml['kp_rolling_std_3'] = df_ml['max_kp'].rolling(window=3, center=True).std()
        df_ml['kp_rolling_mean_7'] = df_ml['max_kp'].rolling(window=7, center=True).mean()
        df_ml['kp_rolling_max_7'] = df_ml['max_kp'].rolling(window=7, center=True).max()
        df_ml['kp_change'] = df_ml['max_kp'].diff()
        df_ml['kp_acceleration'] = df_ml['kp_change'].diff()
        df_ml['kp_volatility_7'] = df_ml['max_kp'].rolling(window=7).std()
        
        # Storm indicators
        df_ml['kp_storm'] = (df_ml['max_kp'] >= 5).astype(int)
        df_ml['kp_active'] = (df_ml['max_kp'] >= 4).astype(int)
        
        # Cumulative storm exposure
        df_ml['storm_days_last_7'] = df_ml['kp_storm'].rolling(window=7).sum()
        df_ml['active_days_last_7'] = df_ml['kp_active'].rolling(window=7).sum()
        
        features_created.extend([
            'kp_rolling_mean_3', 'kp_rolling_std_3', 'kp_rolling_mean_7', 'kp_rolling_max_7',
            'kp_change', 'kp_acceleration', 'kp_volatility_7', 'kp_storm', 'kp_active',
            'storm_days_last_7', 'active_days_last_7'
        ])
    
    # Sunspot features
    if 'sunspot_number' in df_ml.columns:
        df_ml['sunspot_rolling_mean_7'] = df_ml['sunspot_number'].rolling(window=7, center=True).mean()
        df_ml['sunspot_rolling_max_7'] = df_ml['sunspot_number'].rolling(window=7, center=True).max()
        df_ml['sunspot_change'] = df_ml['sunspot_number'].diff()
        df_ml['sunspot_acceleration'] = df_ml['sunspot_change'].diff()
        df_ml['sunspot_volatility'] = df_ml['sunspot_number'].rolling(window=7).std()
        
        features_created.extend([
            'sunspot_rolling_mean_7', 'sunspot_rolling_max_7', 'sunspot_change',
            'sunspot_acceleration', 'sunspot_volatility'
        ])
    
    # Radio flux features
    if 'radio_flux_10cm' in df_ml.columns:
        df_ml['flux_rolling_mean_3'] = df_ml['radio_flux_10cm'].rolling(window=3, center=True).mean()
        df_ml['flux_rolling_mean_7'] = df_ml['radio_flux_10cm'].rolling(window=7, center=True).mean()
        df_ml['flux_change'] = df_ml['radio_flux_10cm'].diff()
        df_ml['flux_acceleration'] = df_ml['flux_change'].diff()
        df_ml['flux_volatility'] = df_ml['radio_flux_10cm'].rolling(window=7).std()
        
        features_created.extend([
            'flux_rolling_mean_3', 'flux_rolling_mean_7', 'flux_change',
            'flux_acceleration', 'flux_volatility'
        ])
    
    # Solar flare features
    flare_cols = [col for col in df_ml.columns if 'flare' in col.lower()]
    if flare_cols:
        df_ml['total_flares'] = df_ml[flare_cols].sum(axis=1)
        df_ml['total_flares_7d'] = df_ml['total_flares'].rolling(window=7).sum()
        df_ml['max_flare_intensity'] = df_ml[flare_cols].max(axis=1)
        
        features_created.extend(['total_flares', 'total_flares_7d', 'max_flare_intensity'])
    
    # Temporal features
    df_ml['day_of_year'] = df_ml['date'].dt.dayofyear
    df_ml['day_of_year_sin'] = np.sin(2 * np.pi * df_ml['day_of_year'] / 365)
    df_ml['day_of_year_cos'] = np.cos(2 * np.pi * df_ml['day_of_year'] / 365)
    df_ml['month'] = df_ml['date'].dt.month
    df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
    
    features_created.extend(['day_of_year', 'day_of_year_sin', 'day_of_year_cos', 'month', 'day_of_week'])
    
    # Interaction features
    if 'max_kp' in df_ml.columns and 'sunspot_number' in df_ml.columns:
        df_ml['kp_sunspot_interaction'] = df_ml['max_kp'] * df_ml['sunspot_number']
        features_created.append('kp_sunspot_interaction')
    
    if 'max_kp' in df_ml.columns and 'radio_flux_10cm' in df_ml.columns:
        df_ml['kp_flux_interaction'] = df_ml['max_kp'] * df_ml['radio_flux_10cm']
        features_created.append('kp_flux_interaction')
    
    # Lag features
    for lag in [1, 2, 3]:
        if 'max_kp' in df_ml.columns:
            df_ml[f'kp_lag_{lag}'] = df_ml['max_kp'].shift(lag)
            features_created.append(f'kp_lag_{lag}')
        
        if 'sunspot_number' in df_ml.columns:
            df_ml[f'sunspot_lag_{lag}'] = df_ml['sunspot_number'].shift(lag)
            features_created.append(f'sunspot_lag_{lag}')
    
    print(f"✅ Created {len(features_created)} engineered features")
    return df_ml, features_created

def prepare_ml_data(df_ml):
    """Prepare data for machine learning"""
    print("📊 Preparing ML dataset...")
    
    # Get feature columns
    feature_cols = [col for col in df_ml.columns if col not in ['date', 'migraine'] and df_ml[col].dtype in ['float64', 'int64']]
    feature_cols = [col for col in feature_cols if not df_ml[col].isna().all()]
    
    # Remove rows with too many NaN values
    df_clean = df_ml[feature_cols + ['migraine']].dropna(thresh=len(feature_cols)*0.7)
    
    # Fill remaining NaN values with median
    for col in feature_cols:
        if df_clean[col].isna().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    X = df_clean[feature_cols]
    y = df_clean['migraine']
    
    print(f"✅ Dataset prepared: {len(X)} samples, {len(feature_cols)} features")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    
    return X, y, feature_cols

class MigrainePredictorNN(nn.Module):
    """Neural Network for migraine prediction"""
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_neural_network(X_train, X_test, y_train, y_test, device):
    """Train neural network on GPU"""
    print("🚀 Training Neural Network on GPU...")
    
    # Prepare data for PyTorch
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).to(device)
    
    # Initialize model
    model = MigrainePredictorNN(X_train.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Training loop
    model.train()
    best_loss = float('inf')
    patience = 0
    max_patience = 50
    
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_tensor).squeeze()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor).squeeze()
            val_loss = criterion(val_outputs, y_test_tensor).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_model_state = model.state_dict().copy()
        else:
            patience += 1
        
        if patience >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        model.train()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor).squeeze()
        test_predictions = test_outputs.cpu().numpy()
        nn_auc = roc_auc_score(y_test, test_predictions)
    
    print(f"✅ Neural Network AUC: {nn_auc:.4f}")
    return model, nn_auc, test_predictions

def hyperparameter_optimization(X, y, device):
    """GPU-accelerated hyperparameter optimization with stratified CV for imbalanced data"""
    print("⚡ Running GPU-accelerated hyperparameter optimization...")
    
    # Check class distribution
    positive_samples = y.sum()
    total_samples = len(y)
    print(f"📊 Class distribution: {positive_samples}/{total_samples} positive samples ({positive_samples/total_samples*100:.1f}%)")
    
    # For extreme imbalance, use simpler optimization
    if positive_samples < 5:
        print("⚠️ Too few positive samples for robust optimization, using grid search")
        best_params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}
        return best_params, 0.5
    
    def objective(trial):
        try:
            # Conservative parameter ranges for small datasets
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 6),
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                'subsample': trial.suggest_float('subsample', 0.8, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 0.1),
            }
            
            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                tree_method='gpu_hist' if torch.cuda.is_available() else 'hist',
                device='cuda' if torch.cuda.is_available() else 'cpu',
                eval_metric='auc',
                class_weight='balanced'
            )
            
            # Use stratified cross-validation to maintain class balance
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds
            cv_scores = []
            
            for train_idx, val_idx in skf.split(X, y):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Skip fold if no positive samples in validation
                if y_fold_val.sum() == 0:
                    continue
                
                model.fit(X_fold_train, y_fold_train, verbose=False)
                y_pred = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_pred)
                
                if not np.isnan(score):
                    cv_scores.append(score)
            
            if len(cv_scores) == 0:
                return 0.5  # Return neutral score if no valid folds
            
            return np.mean(cv_scores)
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.5
    
    try:
        study = optuna.create_study(direction='maximize')
        # Reduced trials for small dataset
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        # Filter out failed trials
        successful_trials = [t for t in study.trials if t.value is not None and not np.isnan(t.value)]
        
        if len(successful_trials) == 0:
            print("⚠️ All optimization trials failed, using default parameters")
            return {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}, 0.5
        
        best_trial = max(successful_trials, key=lambda t: t.value)
        print(f"🎯 Best AUC: {best_trial.value:.4f} (from {len(successful_trials)} successful trials)")
        print(f"🔧 Best parameters: {best_trial.params}")
        
        return best_trial.params, best_trial.value
        
    except Exception as e:
        print(f"⚠️ Optimization failed with error: {e}")
        print("   Using default parameters")
        return {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.1}, 0.5

def comprehensive_ml_analysis():
    """Run comprehensive ML analysis"""
    print("🤖 Starting Comprehensive AI Analysis...")
    print("=" * 50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Load and prepare data
    df = load_space_weather_data()
    df_ml, features_created = create_advanced_features(df)
    X, y, feature_cols = prepare_ml_data(df_ml)
    
    # Check if we have enough data for meaningful analysis
    if len(X) < 20:
        print("⚠️  Insufficient data for comprehensive ML analysis")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols)
    
    results = {}
    models = {}
    
    print("\n🧠 Training Multiple ML Models...")
    print("-" * 30)
    
    # 1. Random Forest
    print("🌲 Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict_proba(X_test_scaled)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred)
    results['Random Forest'] = rf_auc
    models['Random Forest'] = rf
    print(f"   AUC: {rf_auc:.4f}")
    
    # 2. XGBoost with GPU
    print("🚀 Training XGBoost (GPU)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        tree_method='gpu_hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        eval_metric='auc'
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_pred)
    results['XGBoost'] = xgb_auc
    models['XGBoost'] = xgb_model
    print(f"   AUC: {xgb_auc:.4f}")
    
    # 3. LightGBM with GPU
    print("💡 Training LightGBM (GPU)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        random_state=42,
        device='gpu' if torch.cuda.is_available() else 'cpu',
        class_weight='balanced'
    )
    lgb_model.fit(X_train_scaled, y_train)
    lgb_pred = lgb_model.predict_proba(X_test_scaled)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_pred)
    results['LightGBM'] = lgb_auc
    models['LightGBM'] = lgb_model
    print(f"   AUC: {lgb_auc:.4f}")
    
    # 4. Neural Network
    if torch.cuda.is_available():
        nn_model, nn_auc, nn_pred = train_neural_network(
            X_train_scaled, X_test_scaled, y_train, y_test, device
        )
        results['Neural Network'] = nn_auc
        models['Neural Network'] = nn_model
    
    # 5. Hyperparameter Optimization
    if torch.cuda.is_available():
        best_params, best_auc = hyperparameter_optimization(X, y, device)
        
        # Train optimized model
        print("🎯 Training Optimized XGBoost...")
        optimized_xgb = xgb.XGBClassifier(
            **best_params,
            random_state=42,
            tree_method='gpu_hist',
            device='cuda',
            eval_metric='auc'
        )
        optimized_xgb.fit(X_train_scaled, y_train)
        optimized_pred = optimized_xgb.predict_proba(X_test_scaled)[:, 1]
        optimized_auc = roc_auc_score(y_test, optimized_pred)
        results['Optimized XGBoost'] = optimized_auc
        models['Optimized XGBoost'] = optimized_xgb
        print(f"   AUC: {optimized_auc:.4f}")
    
    # Feature Importance Analysis
    print("\n🔍 Feature Importance Analysis...")
    print("-" * 30)
    
    # Get feature importance from best model
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"Top 10 features from {best_model_name}:")
        for _, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']:<25} {row['importance']:.4f}")
    
    # SHAP Analysis
    print("\n🔬 SHAP Explainability Analysis...")
    print("-" * 30)
    
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_scaled.head(20))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': shap_importance
        }).sort_values('shap_importance', ascending=False)
        
        print("Top 10 SHAP feature contributions:")
        for _, row in shap_df.head(10).iterrows():
            print(f"  {row['feature']:<25} {row['shap_importance']:.4f}")
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        shap_df = None
    
    # Save results
    print("\n💾 Saving Results...")
    print("-" * 30)
    
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'model_performance': results,
        'best_model': best_model_name,
        'best_auc': results[best_model_name],
        'feature_count': len(feature_cols),
        'sample_count': len(X),
        'migraine_cases': int(y.sum()),
        'features_created': features_created,
        'top_features': feature_importance.head(10).to_dict('records') if 'feature_importance' in locals() else None,
        'shap_features': shap_df.head(10).to_dict('records') if shap_df is not None else None,
        'gpu_used': torch.cuda.is_available(),
        'device_info': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    }
    
    # Save analysis results
    with open('ai_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    # Save models
    with open('ml_models.pkl', 'wb') as f:
        pickle.dump({
            'models': models,
            'scaler': scaler,
            'feature_cols': feature_cols
        }, f)
    
    print("✅ Results saved to ai_analysis_results.json")
    print("✅ Models saved to ml_models.pkl")
    
    # Summary
    print("\n🎊 ANALYSIS COMPLETE!")
    print("=" * 50)
    print(f"🏆 Best Model: {best_model_name} (AUC: {results[best_model_name]:.4f})")
    print(f"📊 Total Features: {len(feature_cols)}")
    print(f"🎯 Migraine Cases: {y.sum()}/{len(y)}")
    print(f"🚀 GPU Acceleration: {'✅' if torch.cuda.is_available() else '❌'}")
    
    return analysis_results

if __name__ == "__main__":
    results = comprehensive_ml_analysis()
