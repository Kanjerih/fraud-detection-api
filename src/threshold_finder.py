# +
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

def optimize_threshold(model, X_test_scaled, y_test):
    print("--> Evaluating Model Performance...")
    
    # Get raw probabilities
    y_probs = model.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Find the threshold maximizing the F1-Score balance
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    print(f"\n[Optimal Decision Boundary Threshold Selected: {best_threshold:.4f}]")
    
    # Evaluate at our newly optimized boundary
    ypred_custom = (y_probs >= best_threshold).astype(int)
    print("\nClassification Report at Optimized Boundary:")
    print(classification_report(y_test, ypred_custom))
    
    # Plot operational matrix
    conf_matrix = confusion_matrix(y_test, ypred_custom)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title("Confusion Matrix at Optimized Threshold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    
    # CHANGED: Create a reports directory and save the file instead of blocking the terminal
    os.makedirs("reports", exist_ok=True)
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close() # Closes the figure container cleanly out of memory
    print("--> Confusion matrix visualization saved to reports/confusion_matrix.png")
    
    return float(best_threshold)
