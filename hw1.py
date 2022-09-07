#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 13:29:26 2022

@author: marilinaorihuela
"""
#%%
import pandas as pd
import numpy as np
#%%

print(f'1. Numpy version: {np.__version__}\n')      #already insttalled

df = pd.read_csv('data/data.csv')

print(f'2. Dataset rows: {len(df.index)}\n')

count_make = df['Make'].value_counts()
popular = count_make.head(3)        #'Chevrolet', 'Ford', 'Volkswagen'

print(f'3. Most popular car manufacterers: {popular.index[0]}, {popular.index[1]} , {popular.index[2]}\n')

df_audi = df[df['Make'] == 'Audi'].nunique() #34 models

print(f'4. Unique Audi car models: {df_audi[1]}\n')

dm = df.isnull().sum()

print(f'5. Colums with missing values: \n{dm[dm>0]}\n')

median = df['Engine Cylinders'].median()            #6 
frequent = df['Engine Cylinders'].mode()            #4.0
new_df = df['Engine Cylinders'].fillna(4.0)
new_median = new_df.median()                        #6

print(f'6. Median value of "Engine Cylinders" :{median}')
print(f'Most frequent value: {frequent[0]}')
print(f'New median value of "Engine Cylinders" :{new_median}\n')

df_lotus = df[df['Make'] == 'Lotus']            #Select Lotus
columns = ['Engine HP','Engine Cylinders']      
df_lotus = df_lotus[columns]                    #Select columns
df_lotus = df_lotus.drop_duplicates()           #9 rows


X = np.array(df_lotus)                          #(9,2)
XTX  = X.T.dot(X)                               #(2,2)

XTX_invert = np.linalg.inv(XTX)                 #Invert XTX 

y = np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])

w = XTX_invert.dot(X.T).dot(y)              #array([  4.59494481, -63.56432501])

print(f'7. Value of the first element of w: {w[0]}')






