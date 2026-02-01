# utils.py

"""
Utility function for Fraud Detection Project
It has evaluation metrics, plotting functions and result tracking

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score, 
                             confusion_matrix, roc_curve, precision_recall_curve)
from config import SEED  # FOR REPRODUCIBILITY

# Result Tracker
results = []  # Fixed: Changed from 'result' to 'results'

def evaluate_model(y_true, y_pred, y_pred_proba=None, model_name="Unnamed Model"):
    """
    Evaluate a model and return a dictionary of metrics.

    Parameters:
    y_true: Actual labels
    y_pred: Predicted labels
    y_pred_proba: Predicted probabilities (for ROC-AUC and PR-AUC)
    model_name: Name of the model for tracking

    Returns:
    Dictionary of evaluation metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'runtime': None  # To be filled in later
    }

    # Adding probability-based metrics if available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """
    Plot a confusion matrix

    Parameters: 
    y_true: Actual labels
    y_pred: Predicted labels
    model_name: Name of the model for the title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot ROC curve

    Parameter: 
    y_true: Actual labels
    y_pred_proba: Predicted probabilities
    model_name: Name of the model for the title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def plot_pr_curve(y_true, y_pred_proba, model_name="Model"):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    y_true: Actual labels
    y_pred_proba: Predicted probabilities
    model_name: Name of the model for the title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f'pr_curve_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def plot_feature_importance(feature_importances, feature_names, model_name="Model", top_n=20):
    """
    Plot feature importances.

    Parameters:
    feature_importances: Array of feature importances
    feature_names: Names of the features
    model_name: Name of the model for the title
    top_n: Number of top features to show
    """
    # Creating a DataFrame for easier manipulation
    feat_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances,
    })

    # Sort by importance and take top_n
    feat_imp_df = feat_imp_df.sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feat_imp_df)), feat_imp_df['importance'], align='center')
    plt.yticks(range(len(feat_imp_df)), feat_imp_df['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Feature Importances - {model_name}')
    plt.gca().invert_yaxis()  # Most Important at the top
    plt.tight_layout()
    plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.show()

def timer(func):
    """
    Decorator to measure function execution time.

    Parameters:
    func: Function to time

    Returns: 
    Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result, end_time - start_time
    return wrapper

def add_to_results(metrics_dict):
    """
    Add model result to the global result tracker.

    Parameters:
    metrics_dict: Dictionary of model metrics
    """
    global results
    results.append(metrics_dict)

def display_results():
    """
    Display all results in a formatted table
    """
    if not results:
        print("No results to display.")
        return
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string())  # Using to_string() for better formatting

def save_results(filename="model_results.csv"):
    """
    Save results to a CSV file.

    Parameters: 
    filename: Name of the CSV file
    """
    if not results:
        print("No results to save")
        return
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")