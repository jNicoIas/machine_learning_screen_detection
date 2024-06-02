#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:30:59 2024

@author: jnicolas
"""

# %% Import Libraries

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import sys
import csv
from io import StringIO
import os


class SensordataProcessor:
    def __init__(self, file_path):
        # Construct the full file path using the "sensor_data" folder
        full_file_path = os.path.join("sensor_data", file_path)
        
        # Read the CSV file
        raw_data = pd.read_csv(full_file_path)
        
        # Separate x and y variables 
        self.y = raw_data.pop('Exposure')
        self.x = raw_data
    
    def preprocess_data(self):
        # Transform Categorical data into to Numerical Values --------------------------------------------------------------------------------
        labelEncoder_y = LabelEncoder()
        self.y = labelEncoder_y.fit_transform(self.y)        # Dynamic = 0, No Screen = 1, Static = 2


        # Absolute values
        self.sensordata = self.x.abs()                   # Takes absolute value


        # Median Filter  --------------------------------------------------------------------------------

        #Light
        MedianLight = self.sensordata['Light'].rolling(3).median()       # median filter for light
        self.sensordata.insert(4,'LightFiltered', MedianLight)           # appends filtered Light data
        #Red
        MedianR = self.sensordata['R'].rolling(3).median()              # median filter for light
        self.sensordata.insert(5,'RedFiltered', MedianR)                # appends filtered Light data
        #Green
        MedianG = self.sensordata['G'].rolling(3).median()              # median filter for light
        self.sensordata.insert(6,'GreenFiltered', MedianG)              # appends filtered Light data
        #Blue
        MedianB = self.sensordata['B'].rolling(3).median()              # median filter for light
        self.sensordata.insert(7,'BlueFiltered', MedianB)               # appends filtered Light data


        # Imputedata --------------------------------------------------------------------------------
        impute_x = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
        Filtereddata = impute_x.fit_transform(self.sensordata) # Filtered and Imputed data

        # Conversion of array to frame --------------------------------------------------------------------------------
        SegmentedFiltereddata = Filtereddata[:,4:8]             # Removed the raw data
        self.sensordata = pd.DataFrame(SegmentedFiltereddata)
        self.sensordata.columns =['Light', 'R', 'G', 'B']
    
    def compute_features(self):
        # Feature Computation ---------------------------------------------------------------------------------------------
        
        sensordata = self.sensordata
        data = sensordata[["Light","R","G","B"]]                #Transfers sensordata to data

        data['R/L'] = sensordata['R']/sensordata['Light']
        data['(L-G)/G'] = (sensordata['Light']-sensordata['G'])/sensordata['G']
        data['(L-R)/R'] = (sensordata['Light']-sensordata['R'])/sensordata['R']
        data['sqrt(B)/sqrt(L)'] = np.sqrt(sensordata['B'])/np.sqrt(sensordata['Light'])
        data['sqrt(R)/sqrt(L)'] = np.sqrt(sensordata['R'])/np.sqrt(sensordata['Light'])
        data['sqrt(G)/sqrt(L)'] = np.sqrt(sensordata['G'])/np.sqrt(sensordata['Light'])
        data['log(R)/log(C)'] = np.log10(sensordata['R'])/np.log10(sensordata['Light'])
        data['log(G)/log(C)'] = np.log10(sensordata['G'])/np.log10(sensordata['Light'])
        data['log(B)/log(C)'] = np.log10(sensordata['B'])/np.log10(sensordata['Light'])

        #Ratio of Present and Past + Ratio of Past and Present --------------------------------------------------------------------------------

        data["Light L2H"] = data["Light"] / data["Light"].shift(1)
        data["Red L2H"] = data["R"] / data["R"].shift(1)
        data["Green L2H"] = data["G"] / data["G"].shift(1)
        data["Blue L2H"] = data["B"] / data["B"].shift(1)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Imputedata --------------------------------------------------------------------------------
        from sklearn.impute import KNNImputer   #imputer that look at neighbors to generate value for NaN
        impute_x = KNNImputer(missing_values=np.nan, n_neighbors=5, weights='uniform', metric='nan_euclidean')
        datax = impute_x.fit_transform(data) # Filtered and Imputed data


        # Converts dataframe to Array 
        y_train = self.y
        x_train = datax[:, [2,3,4,5,6,7,8,9,10,11,12,13,15,16]] 

        # Split between training and testing data --------------------------------------------------------------------------------

        x_train, x_test, self.y_train, self.y_test = train_test_split(x_train,y_train,train_size=0.8, random_state=0) # shuffled

        # TO PERFORM FEATURE SCALING  --------------------------------------------------------------------------------
        self.standard_scaler_x = StandardScaler() # created an object: standard_scaler_x
        self.x_train = self.standard_scaler_x.fit_transform(x_train) 
        self.x_test = self.standard_scaler_x.fit_transform(x_test)
    
    def machine_learning(self):
        # Fit
        self.random_forest_classifier = RandomForestClassifier(max_depth=None, random_state=0)
        self.random_forest_classifier.fit(self.x_train,self.y_train)

        #---------------------------------------
        # Predict output of the testing dataset
        self.y_predict = self.random_forest_classifier.predict(self.x_test)

        # Performance Metrics of SVM Model with DEFAULT PARAMETERS
        print("CONFUSION MATRIx:")
        print(confusion_matrix(self.y_test, self.y_predict))

        classification_accuracy = accuracy_score(self.y_test, self.y_predict)
        print('Classification Accuracy: %.4f'
              % classification_accuracy)
        print('')

        print("CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, self.y_predict, digits=4))
    
    def evaluate_performance(self):
        # To apply K-fold cross validation for the model's performance
        k_fold = KFold(n_splits=10)

        # To validate K-fold accuracy
        self.x_feature_scaled = self.standard_scaler_x.fit_transform(self.x)
        accuracies = cross_val_score(estimator=self.random_forest_classifier,X=self.x_feature_scaled, y=self.y, cv=k_fold,scoring='accuracy')
        accuracies_average = accuracies.mean()
        print('K-Fold Average Accuracy: %.4f' %accuracies_average)
        accuracies_variance = accuracies.std()
        print('K-Fold Variance: %.4f' %accuracies_variance)


        # For the Classification Accuracy (Holdout)
        classification_accuracy=accuracy_score(self.y_test,self.y_predict)
        print('Classfication Accuracy: %.4f'
              %classification_accuracy)
        print(' ')


    
    def compute_screentime(self):

       dynamictime_predict = ((self.y_predict == 0).sum())*0.208470175
       statictime_predict = ((self.y_predict == 2).sum())*0.20685915
       noscreentime_predict = ((self.y_predict == 1).sum())*0.21502807
                               
       dynamictime_test = ((self.y_test == 0).sum())*0.208470175
       statictime_test = ((self.y_test == 2).sum())*0.20685915
       noscreentime_test = ((self.y_test == 1).sum())*0.21502807

       dynamictime_total = ((self.y == 0).sum())*0.208470175
       statictime_total = ((self.y == 2).sum())*0.20685915
       noscreentime_total = ((self.y == 1).sum())*0.21502807


       total_predict = dynamictime_predict + statictime_predict + noscreentime_predict
       total_theotime = dynamictime_test + statictime_test + noscreentime_test
       total_time = dynamictime_total + statictime_total + noscreentime_total
       # time_ratio = total_theotime/total_time                    
       time_ratio = abs((total_predict-total_theotime)/total_theotime)
                     
       Screentime = [[dynamictime_predict, dynamictime_test],[statictime_predict,statictime_test],[noscreentime_predict,noscreentime_test]]         
       Screentime = pd.DataFrame(Screentime, columns=['Experimental (s)', 'Actual (s)'])     
       Screentime['Error (s)'] = abs(Screentime['Experimental (s)'] - Screentime['Actual (s)'])
       Screentime['Percent Error (%)'] = (Screentime['Error (s)'] / Screentime['Actual (s)'])*100
       print(Screentime)


       # print("Total time:", total_time)
       print("Total predicted time:", total_predict)
       print("Total test time:", total_theotime)
       print("Percent Error:", time_ratio*100)
       
    def export_prints_to_csv(self, csv_file_path):
        # Create a buffer to capture the printed output
        stdout_backup = sys.stdout
        sys.stdout = buffer = StringIO()
        
        # Call methods containing print statements
        self.preprocess_data()
        self.compute_features()
        self.machine_learning()
        self.evaluate_performance()
        self.compute_screentime()
        
        # Restore the original stdout and extract the captured output
        sys.stdout = stdout_backup
        buffer.seek(0)
        printed_output = buffer.getvalue().splitlines()
        
        # Create the "results" folder if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Write the printed output to a CSV file in the "results" folder
        with open(os.path.join("results", csv_file_path), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Printed Output'])
            writer.writerows([[line] for line in printed_output])


# if __name__ == "__main__": 

#     processor = sensordataProcessor('S2-3200-GL5528-40CM.csv')
#     processor.export_prints_to_csv('try.csv')