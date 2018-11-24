from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc(pred, real, etiq, color='b'):
    fpr, tpr, thresholds = roc_curve(real, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, color=color, label=etiq + ' (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')