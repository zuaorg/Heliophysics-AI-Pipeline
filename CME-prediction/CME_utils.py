from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd


def calculate_metrics(y_test, y_pred, y_pred_prob, algorithm, time_window):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # Recall
    recall = recall_score(y_test, y_pred)
    print("Recall:", recall)

    # Precision
    precision = precision_score(y_test, y_pred)
    print("Precision:", precision)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # True Skill Score (TSS)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    T = len(y_test)  # Total labels
    P = np.sum(y_test == 1)  # Positive labels
    N = np.sum(y_test == 0)  # Negative labels
    tss = (TP / (TP + FN)) - (FP / (FP + TN))
    print("True Skill Score (TSS):", tss)

    # Heidke Skill Score (HSS)
    hss = (2 * (TP * TN - FP * FN)) / ((TP + FP) * (FP + TN) + (TP + FN) * (TN + FN))
    print("Heidke Skill Score (HSS):", hss)

    # AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    print("AUC:", auc)

    # Balanced Accuracy (BACC) and WAUC
    bacc = (recall + precision) / 2

    # Brier Score (BSC) and Brier Skill Score (BSSC)
    bsc = np.mean((y_pred_prob - y_test) ** 2)
    bssc = 1 - (bsc / np.mean((y_test - np.mean(y_test)) ** 2))  # Brier Skill Score

    # Prepare dictionary with the metrics to save
    p_dic = {
        'Algorithm': [algorithm],
        'TW': [time_window],
        'T': [T],
        'P': [P],
        'N': [N],
        'TP': [TP],
        'TN': [TN],
        'FP': [FP],
        'FN': [FN],
        'ACC': [accuracy],
        'BACC': [bacc],
        'Pre': [precision],
        'Rec': [recall],
        'TSS': [tss],
        'HSS': [hss],
        'AUC': [auc],
        'BSC': [bsc],
        'BSSC': [bssc]
    }

    # Creating DataFrame
    cols = ['Algorithm', 'TW', 'T', 'P', 'N', 'TP', 'TN', 'FP', 'FN', 'ACC', 'BACC', 'Pre', 'Rec',
            'TSS', 'HSS', 'AUC', 'BSC', 'BSSC']
    p_df = pd.DataFrame(p_dic, columns=cols)

    # Save the performance metrics to file
    file = str(algorithm) + '-' + str(time_window) + '-performance-metrics.csv'
    print('Saving the performance metrics to file:', file)
    p_df.to_csv(file, index=False)

    print(p_df)