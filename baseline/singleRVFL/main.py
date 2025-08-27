from argparse import ArgumentParser
import math
import csv
import os
import pathlib
from pprint import pprint
import math

import singlelayer 

#from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.datasets import load_wine


# Path to dataset 
DATAPATH = ''
folders = os.listdir(PATH)
RES = []


def run(): 
    
    # Hyper parameter 1/C 
    C_list = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]  
    progress = 0

    # Load the wine dataset into a DataFrame
    data_all = load_wine(as_frame=True)
    df_all = data.frame

    print(df_all.head())

    # Display the shape of the DataFrame
    print("Shape of the Wine DataFrame:", df_all.shape)

    kf = KFold(n_splits=4, shuffle=False)

    for C in C_list:
        accuracy = 0
        for train_index, test_index in kf.split(wine_df): 

            beta, A, B = singlelayer(df_all[train_index], df_all.target[train_index], L=100, C=C)
                                                    
            y_valid = predict(df_all[test_index], beta, A, B)
            accuracy += evaluation(y_valid, df_all[test_index])
            
        RES += [str(accuracy/4) + '   C = ' + str(C)]

    
    RES = np.array(RES)
    np.savetxt("RVFL_acc_100.txt", RES, fmt='%s', delimiter=',')






