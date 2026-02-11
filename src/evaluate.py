from sklearn.metrics import roc_auc_score, classification_report


def evaluate_model(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)

    print("ROC-AUC:", roc_auc)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return roc_auc
