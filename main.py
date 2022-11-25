# Importing all the necessary modules
import sys
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")
#DATA ANALYSIS
#1.EXPLORATION
df = pd.read_csv("Life Expectancy Data.csv")
print(df.head())
print("---------------------------------------------------------------------")

# I want to create a life_expectancy encoder function that can create a range of values for different life expectancies
def life_expectancy_encoder(x:np.float64):
    if 35 <= x < 40:
        return 1
    elif 40 <= x < 45:
        return 2
    elif 45 <= x < 50:
        return 3
    elif 50 <= x < 55:
        return 4
    elif 55 <= x < 60:
        return 5
    elif 60 <= x < 65:
        return 6
    elif 65 <= x < 70:
        return 7
    elif 70 <= x < 75:
        return 8
    elif 75 <= x < 80:
        return 9
    elif 80 <= x < 85:
        return 10
    elif 85 <= x < 90:
        return 11

df.rename(columns={'Life expectancy ':'Life expectancy','Adult Mortality':'Adult Mortality','infant deaths':'infant deaths',
                   'Hepatitis B':'Hepatitis B','percentage expenditure':'percentage expenditure','Measles ':'Measles',
                   ' BMI ':'BMI','under-five deaths ':'under-five deaths','Total expenditure':'Total expenditure',
                  'Diphtheria ':'Diphtheria',' HIV/AIDS':'HIV/AIDS',' thinness  1-19 years':'thinness 1-19 years',
                  ' thinness 5-9 years':'thinness 5-9 years','Income composition of resources':'Income composition of resources'},inplace=True)
print(df.columns)
print("---------------------------------------------------------------------")

# Here I am trying to convert float values to int so that I  can apply classification models
df["Life expectancy"] = df["Life expectancy"].apply(life_expectancy_encoder)
print(df.info())
print("---------------------------------------------------------------------")
print(df.head())
print("---------------------------------------------------------------------")

# checking the null values of Life expectancy and deleting them
bool1 = pd.isnull(df['Life expectancy'])
df[bool1]
df = df.dropna(subset=['Life expectancy'])
print(df.describe())
print("---------------------------------------------------------------------")

#Cleaning
# we are using the analysis to find out the best strategies for data cleaning and
print(df.isnull().sum())
print("---------------------------------------------------------------------")

#Plotting
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
print("---------------------------------------------------------------------")

# subplots
plt.figure(figsize=(12,18))
plt.subplots_adjust(hspace=0.5)
i = 1
for col_name in df.drop("Country", axis=1).columns:
    plt.subplot(10,4, i)
    sns.histplot(data=df, x=col_name, kde=True,  multiple='layer', alpha=0.5, palette='viridis')
    i += 1
print("---------------------------------------------------------------------")


#1.OUTLIERS

# using k-score method

iqr = df.quantile(0.75) - df.quantile(0.25)
lower = df.quantile(0.25) - 1.5*iqr
upper = df.quantile(0.75) + 1.5*iqr

print("Number of outliers")
print(((df < lower) + (df > upper)).sum())
print("---------------------------------------------------------------------")

iqr = df.quantile(0.75) - df.quantile(0.25)
lower = df.quantile(0.25) - 3*iqr
upper = df.quantile(0.75) + 3*iqr

print("Number of outliers")
print(((df < lower) + (df > upper)).sum())
print("---------------------------------------------------------------------")

#2. Imputing and Encoding
# Label encoding for status of countries
# Label Encoding on the Package attribute
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
#Assigning numerical values and storing in another column
df['Status'] = labelencoder.fit_transform(df['Status'])
# reverse function because it assigned opposite value to "developing" and "developed"
def reverse(x):
    if x == 0:
        return 1
    elif x == 1:
        return 0
df["Status"] = df['Status'].apply(reverse)
print(df.describe())
print("---------------------------------------------------------------------")

df = df.drop('Country', axis = 1)
from sklearn.impute import SimpleImputer
simple_imputer = SimpleImputer(strategy='median')

cols_nan = df.columns[df.isnull().sum() > 0]
simple_imputer.fit(df)
df[cols_nan] = simple_imputer.fit_transform(df[cols_nan])
print(df.describe())
print("---------------------------------------------------------------------")

print(df.isnull().sum())
print("---------------------------------------------------------------------")

#Step 3: Skewness
from scipy.stats import skew
print("Skewness scores for all columns:")
for column in df.columns:
    print(f"{column} : {skew(df[column])}")
print("---------------------------------------------------------------------")


#Step 4: Feature Selection
#Pearson's Correlation Test
# Heatmap after all changes in the data
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)
df.corr()['Life expectancy'].sort_values(ascending=False)
print("---------------------------------------------------------------------")

#Observations:

#The following data points have a very high correlation:

#Under five deaths & Infant deaths (0.99)
#GDP & percentage expenditure (0.7)
#Dipteria & Polio (0.67)
#Thinness 5-9 years & Thinness 1-19 years (0.94)
#Income compostion of resources & Schooling (0.8)

#Information gain
# example of mutual information feature selection for numerical input data
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot
import matplotlib as plt

X = df.drop(columns = ["Life expectancy"], axis = 1) #feature matrix
y = df["Life expectancy"] #target

fs = SelectKBest(score_func=mutual_info_classif, k='all')
# learn relationship from training data
fs.fit(X, y)
# transform train input data
X_train_fs = fs.transform(X)
features = X.columns
print("The feature scores generated using Information Gain method are: ")
features_map = []
for i in range(len(fs.scores_)):
    features_map.append([features[i], float(fs.scores_[i])])
features_map = sorted(features_map, key=lambda x:x[1], reverse = True)
for i in range(len(fs.scores_)):
	print('Feature %s: %f' % (features_map[i][0], features_map[i][1]))
# plot the scores
pyplot.figure(figsize=(50,20))
pyplot.bar([i for i in features], fs.scores_)
pyplot.show()