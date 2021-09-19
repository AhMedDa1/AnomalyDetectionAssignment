import os
import sys
import argparse
import pickle

import pandas as pd

import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn import metrics

from sklearn.metrics import silhouette_score, silhouette_samples


import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
#import seaborn as sns
import matplotlib as mpl

import numpy as np
from scipy.spatial.distance import cdist

import random
random.seed(183703)

# used to check the value intering the silh parser
def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue

   
def dataProcess(df, scaler = None):

    att_columns = ['att1', 'att2', 'att3', 'att4', 'att5', 'att6']
    if df.shape[1] != 6:
      df = df[att_columns]
    num_columns = [column for column in df.columns if df[column].dtype in ["int64", "float64"] ]
    num_X_train = df[num_columns]


    
    #process numerical fetures (scale )
    if scaler is None:
        scaler = MinMaxScaler().fit(num_X_train)
    
    scale_num_X_train = pd.DataFrame(scaler.transform(num_X_train),
        columns = num_X_train.columns,
        index = num_X_train.index)
        
    return scaler, scale_num_X_train
    
    
def predict_anomal(model, data, percentile = 98):
    
    centroids = model.cluster_centers_
    # points array will be used to reach the index easy
    points = np.empty((0,len(data.iloc[0]) + 1 ), float)
    # distances will be used to calculate outliers
    distances = np.empty((0,2), float)
    
    clusters=model.fit_predict(data)
    # scale_num_X_train["cluster"] = clusters
    rest_X_train = data.reset_index()

    # getting points and distances
    for i, center_elem in enumerate(centroids):
        array = rest_X_train[clusters == i].drop(['index'], axis =1)
        # cdist is used to calculate the distance between center and other points
        distances = np.append(distances, cdist([center_elem],array, 'euclidean')) 
        points = np.append(points, rest_X_train[clusters == i], axis=0)
    points = pd.DataFrame(points, columns = rest_X_train.columns )
    points["index"] = points["index"].astype(int)
    points.set_index('index', inplace =True)
    
    # getting outliers whose distances are greater than some percentile
    outliers = points[distances > np.percentile(distances, percentile)]
    points["predict"] = distances > np.percentile(distances, percentile)
    
    answer = points["predict"]
    # answer.set_index("index")
    # answer.cloumns.name = None
    return answer


## this part of code is mostly copyed from the book "hands-on Machin learing"
def silh_score_plot( X_train, k):
    
    """this part of code is mostly copyed from 
    the book "hands-on Machin learing" """
    
    k = k
    model = KMeans(n_clusters= k).fit(X_train)
    y_pred = model.predict(X_train)
    
    silh_score = silhouette_score(X_train, y_pred)
    
    silhouette_coefficients = silhouette_samples(X_train, y_pred)
    
    X = X_train
    silhouette_scores = silh_score
    plt.figure(figsize=(18, 11))
    padding = len(X) // 30
    pos = padding
    ticks = []
    for i in range(k):
        coeffs = silhouette_coefficients[y_pred == i]
        coeffs.sort()
    
        color = mpl.cm.Spectral(i / k)
        plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ticks.append(pos + len(coeffs) // 2)
        pos += len(coeffs) + padding
        
    plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
    plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
    
    plt.ylabel("Cluster")
    plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    plt.axvline(x=silhouette_scores, color="red", linestyle="--")
    plt.title("$k={}$".format(k), fontsize=16)
    
    plt.show()
    
    
parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('--v', action='store_true') ### print the performanc metrecs of the model in the cmd

#silhoute plot for a particluer K value
parser.add_argument('--sil', type=check_positive,
help = "to plot silhout score put k = for example --sil 5 ")
## save the model in pikl format after the path is spasified
## will be saved as "saved_model.csv"
parser.add_argument('--s',
                       metavar='data_path',
                       type=str,
                       help='the path to save the model in')
                       
parser.add_argument('--l',
                       metavar='data_path',
                       type=str,
                       help='the path to the model you want to load (must be .pkl)')


parser.add_argument('-Data',
                       metavar='data_path',
                       default="datasetAssignment2.csv",
                       type=str,
                       help='the path to the training data')
      
 ### return a csv file named [resluts.csv] countain coulmns named index, predict                        
parser.add_argument('-I',
                       metavar = 'test_data',
                       type = str,
                       help = 'the path to the file you want to cluster(must be .CSV)')


#parser.add_argument('--bar', nargs='*', help='set the columns numbers', default=["2","3"])

args = parser.parse_args()


print(args.Data)
accuracy_flag =args.v




output_path = args.I

input_path = args.Data
save_model = args.s
load_model = args.l
k_silh = args.sil

if input_path:
  if not os.path.isfile(input_path):
      print('The path specified does not exist')
      sys.exit()                    
  else:
      df = pd.read_csv(input_path)
      
      X = df.drop(["label"], axis = 1)
      y = df["label"]
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
      
      scaler , data = dataProcess(X_train )
      _ ,X_test = dataProcess(X_train, scaler = scaler)
      
      model= KMeans(n_clusters = 4)
      model.fit(data) 
    
  
  
  ## import or train new or export  
if save_model:
  if not os.path.isdir(save_model):
      print('The path specified does not exist')
      sys.exit()                    
  else:   
      filename = os.path.join(save_model,"saved_model.pkl")
      imports = (model,scaler)
      pickle.dump( imports , open(filename, 'wb'))


if load_model:
  if not os.path.isfile(load_model):
      print('The imported model path does not exist')
      sys.exit()                    
  else:
          filename = load_model
          model, scale = pickle.load(open(filename, 'rb'))

    


if output_path:
    # print(output_path)
    # try:
        _ ,data = dataProcess(pd.read_csv(output_path),scaler = scaler)
        predicts = predict_anomal(model, data, percentile = 98)
        predicts.to_csv("resluts.csv")
        print("see the file [resluts.csv]")
        
    # except:
    #     print("something whent wrong, pleas spacify the file correctly")

if k_silh:
    silh_score_plot(data, k_silh)

	
	
sys.exit()	