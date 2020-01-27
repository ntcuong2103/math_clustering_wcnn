from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import os
# from sklearn.utils.fixes import signature

import numpy as np

def recall_np(y_true, y_pred):
    a = np.array(y_true >= 0.5).astype(float)
    b = np.array(y_pred >= 0.5).astype(float)

    c = a * b
    return np.sum(c) / np.sum(a)

def precision_np(y_true, y_pred):
    a = np.array(y_true >= 0.5).astype(float)
    b = np.array(y_pred >= 0.5).astype(float)

    c = a * b
    return np.sum(c) / (np.sum(b) + 0.001)


def get_precision_recall(test_folder):
    predict_csv_file = os.path.join(test_folder, 'predict.csv')
    target_csv_file = os.path.join(test_folder, 'target.csv')
    num_classes = 101

    y_pred = np.array(np.loadtxt(predict_csv_file, delimiter=",")).astype(float)[:,:num_classes]
    y_true = np.array(np.loadtxt(target_csv_file, delimiter=",")).astype(float)

    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    average_precision = average_precision_score(y_true, y_pred, average="micro")
    # print (average_precision)


    return precision, recall, average_precision

def plot_all():
    test_folder_dataset1 = "./clustering_experiment/cls_local_attn_multiloss/Khuong/"
    test_folder_dataset2 = "./clustering_experiment/cls_local_attn_multiloss/Sasaki2/0/"
    test_folder_CROHME101 = "./clustering_experiment/cls_local_attn_multiloss/CROHME_101/"

    lines = []
    labels = []

    # dataset 1
    precision, recall, average_precision = get_precision_recall(test_folder_dataset1)

    l, = plt.plot(recall, precision, 'r--')
    labels.append('dataset 1, GAtP (AP={:.2f})'.format(average_precision))
    lines.append(l)

    # dataset 2
    precision, recall, average_precision = get_precision_recall(test_folder_dataset2)

    l, = plt.plot(recall, precision, 'g--')
    labels.append('dataset 2, GAtP (AP={:.2f})'.format(average_precision))
    lines.append(l)

    # dataset 3
    precision, recall, average_precision = get_precision_recall(test_folder_CROHME101)

    l, = plt.plot(recall, precision, 'b--')
    labels.append('CROHME, GAtP (AP={:.2f})'.format(average_precision))
    lines.append(l)


    #------------------------------------------------------------

    test_folder_dataset1 = "./clustering_experiment/cls_local_new/Khuong/"
    test_folder_dataset2 = "./clustering_experiment/cls_local_new/Sasaki/0/"
    test_folder_CROHME101 = "./clustering_experiment/cls_local_new/CROHME_101/"

    # dataset 1
    precision, recall, average_precision = get_precision_recall(test_folder_dataset1)

    l, = plt.plot(recall, precision, 'r-.')
    labels.append('dataset 1, GMP (AP={:.2f})'.format(average_precision))
    lines.append(l)

    # dataset 2
    precision, recall, average_precision = get_precision_recall(test_folder_dataset2)

    l, = plt.plot(recall, precision, 'g-.')
    labels.append('dataset 2, GMP (AP={:.2f})'.format(average_precision))
    lines.append(l)

    # dataset 3
    precision, recall, average_precision = get_precision_recall(test_folder_CROHME101)

    l, = plt.plot(recall, precision, 'b-.')
    labels.append('CROHME, GMP (AP={:.2f})'.format(average_precision))
    lines.append(l)

    lines = [lines[i] for i in [0, 3, 1, 4, 2, 5]]
    labels = [labels[i] for i in [0, 3, 1, 4, 2, 5]]



    fig = plt.gcf()
    # fig.subplots_adjust(bottom=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc='lower left')

    plt.show()


def calc_mAP():
    test_folder_dataset1 = "./clustering_experiment/cls_local_attn_multiloss/Khuong/"
    test_folder_dataset2 = "./clustering_experiment/cls_local_attn_multiloss/Sasaki2/0/"
    test_folder_CROHME_attn = "clustering_experiment/crohme2019/attn_new"
    test_folder_CROHME_gap = "clustering_experiment/crohme2019/gap"
    test_folder_CROHME_gmp = "clustering_experiment/crohme2019/gmp"


    # dataset 3
    precision, recall, average_precision = get_precision_recall(test_folder_CROHME_attn)
    print(average_precision)

    precision, recall, average_precision = get_precision_recall(test_folder_CROHME_gap)
    print(average_precision)
    #
    precision, recall, average_precision = get_precision_recall(test_folder_CROHME_gmp)
    print(average_precision)




if __name__ == '__main__':
    calc_mAP()
