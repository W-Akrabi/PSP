import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf

df = pd.read_csv("data_col.csv")

df = df.loc[:, ['Student Age',
'Scholarship type',
'Additional work',
'Regular artistic or sports activity',
'Total salary if available',
'Weekly study hours',
'Attendance to classes',
'Preparation to midterm exams 1',
'Cumulative grade point average in the last semester (/4.00)',
'GRADE']]

if df.isnull().values.any() == True:
    df.dropna(inplace=True)

train, validation, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])

def scale_dataset(dataframe, oversample=False):
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

train, X_train, y_train = scale_dataset(train, True)
valid, X_valid, y_valid = scale_dataset(validation, False)
test, X_test, y_test = scale_dataset(test, False)

nn_model = 