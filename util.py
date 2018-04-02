import sys
import numpy as np
from sklearn import metrics
import csv
import argparse
import util2

def read_txy_csv(fn):
   data = readCsvFile(fn)
   Xtest = data[:, :3]
   Ytest = data[:, 3][:, np.newaxis]
   return Xtest, Ytest

def read_carmen(fn_gfs):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--logfile",
            default=fn_gfs, #wombot_test_2016-02-05-11-51-14.gfs.log  SimonFewLasers.gfs.log
            help="Logfile in CARMEN format to process"
    )

    args = parser.parse_args()

    # Load data and split it into training and testing data
    train_data, test_data = util2.create_test_train_split(args.logfile, 0)

    train_data = np.asarray( train_data["scans"])

    return train_data

def get_mesh_grid(resolution=1, limits=[-120, 120, -20, 120]):
   x_spaced = np.arange(limits[0], limits[1], resolution )
   y_spaced = np.arange(limits[2], limits[3], resolution)
   xx, yy = np.meshgrid(x_spaced, y_spaced)
   X_plot = np.vstack((xx.flatten(),yy.flatten())).T
   return X_plot

def generate_test_data_everyother(fn, X_all, Y_all, step=2):
   #ith_scan_indx_plus1 = X_all[:, 0] > ith_scan
   #Xtest = X_all[ith_scan_indx_plus1, 1:]
   #Ytest = Y_all[ith_scan_indx_plus1, :]

   for jth_scan in range(1, len(np.unique(X_all[:, 0])), step): #start from 1

      jth_scan_indx = X_all[:, 0] == jth_scan

      X_jth_scan = X_all[jth_scan_indx, :]
      Y_jth_scan = Y_all[jth_scan_indx, :]

      X_jth_0 = X_jth_scan[Y_jth_scan[:,0]==0, :]
      X_jth_1 = X_jth_scan[Y_jth_scan[:,0]==1, :]
      if X_jth_1.shape[0] < X_jth_0.shape[0]: #number of ones are less than number of zeros
          n_X_jth_1 = X_jth_1.shape[0]
          indx = np.random.choice(X_jth_0.shape[0], n_X_jth_1, replace=False)
          X_jth_0 = X_jth_0[indx, :]
          n_X_jth_0 = n_X_jth_1
      else:
          n_X_jth_0 = X_jth_0.shape[0]
          indx = np.random.choice(X_jth_1.shape[0], n_X_jth_0, replace=False)
          X_jth_1 = X_jth_1[indx, :]
          n_X_jth_1 = n_X_jth_0
          print('ouch!', jth_scan)

      X_jth = np.vstack((X_jth_0, X_jth_1))
      Y_jth = np.array([[0]*n_X_jth_0 + [1]*n_X_jth_1]).T

      if jth_scan == 1: #first scan to store
         Xtest = X_jth
         Ytest = Y_jth
      else:
         Xtest = np.vstack((Xtest, X_jth))
         Ytest = np.vstack((Ytest, Y_jth))

   points_labels = np.hstack((Xtest, Ytest))
   np.savetxt(fn, points_labels, delimiter=",")

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    return np.array( dataList )

def conf_matrix(p,labels,names=['1','0'],threshold=.5,show=True):
    """
    Returns error rate and true/false positives in a binary classification problem
    - Actual classes are displayed by column.
    - Predicted classes are displayed by row.
    :param p: array of class '1' probabilities.
    :param labels: array of actual classes.
    :param names: list of class names, defaults to ['1','0'].
    :param threshold: probability value used to decide the class.
    :param show: whether the matrix should be shown or not
    :type show: False|True
    """
    assert p.size == labels.size, "Arrays p and labels have different dimensions."
    decision = np.ones((labels.size,1))
    decision[p<threshold] = 0
    diff = decision - labels
    false_0 = diff[diff == -1].size
    false_1 = diff[diff == 1].size
    true_1 = np.sum(decision[diff ==0])
    true_0 = labels.size - true_1 - false_0 - false_1
    error = (false_1 + false_0)/np.float(labels.size)
    if show:
        print(100. - error * 100,'% instances correctly classified')
        print('%-10s|  %-10s|  %-10s| ' % ('',names[0],names[1]))
        print('----------|------------|------------|')
        print('%-10s|  %-10s|  %-10s| ' % (names[0],true_1,false_0))
        print('%-10s|  %-10s|  %-10s| ' % (names[1],false_1,true_0))
    return error, true_1, false_1, true_0, false_0

def neg_ms_log_loss(true_labels, predicted_mean, predicted_var):
    """
    :param true_labels:
    :param predicted_mean:
    :param predicted_var:
    :return: Neg mean squared log loss (neg the better)
    """

    predicted_var += np.finfo(float).eps #to avoid /0 and log(0)
    smse = 0.5*np.log(2*np.pi*predicted_var) + ((predicted_mean - predicted_var)**2)/(2*predicted_var)

    return np.sum(smse)/len(smse)

def calc_scores(mdl_name, true, predicted, predicted_var=None, time_taken=-11):
   fn = 'outputs/reports/'+ mdl_name+ '.csv'

   predicted_binarized = np.int_(predicted >= 0.5)
   accuracy = np.round(metrics.accuracy_score(true.ravel(), predicted_binarized.ravel()), 3)

   auc = np.round(metrics.roc_auc_score(true.ravel(), predicted.ravel()), 3)

   nll = np.round(metrics.log_loss(true.ravel(), predicted.ravel()), 3)

   if predicted_var is not None:
      neg_smse = np.round(neg_ms_log_loss(true, predicted[0].ravel(), predicted_var.ravel()), 3)
   else:
      neg_smse = -11

   print(mdl_name+': accuracy={}, auc={}, nll={}, smse={}, time_taken={}'.format(accuracy, auc, nll, neg_smse, time_taken))
   #print(metrics.confusion_matrix(true.ravel(), predicted_binarized.ravel()))

   with open(fn,'a') as f_handle:
      np.savetxt(f_handle, np.array([[accuracy, auc, nll, neg_smse, time_taken]]), delimiter=',', fmt="%.3f")
