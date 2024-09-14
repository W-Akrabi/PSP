import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from tensorflow.python.keras.engine import data_adapter


def _is_distributed_dataset(ds):
    return isinstance(ds, data_adapter.input_lib.DistributedDatasetSpec)


data_adapter._is_distributed_dataset = _is_distributed_dataset

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

# Split data into training, validation, and test sets using pandas directly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.6 * len(df))
validation_size = int(0.8 * len(df))

train = df_shuffled.iloc[:train_size]
validation = df_shuffled.iloc[train_size:validation_size]
test = df_shuffled.iloc[validation_size:]


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

y_train = to_categorical(y_train, num_classes=8)
y_valid = to_categorical(y_valid, num_classes=8)
y_test = to_categorical(y_test, num_classes=8)

model = Sequential([
    Dense(64, activation='relu', input_shape=(9,)),  # Adjust input_shape to match number of features
    Dense(32, activation='relu'),  # Hidden layer with 32 units
    Dense(16, activation='relu'),  # Hidden layer with 16 units
    Dense(8, activation='softmax')  # Output layer with 5 units (for 5 classes)
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(X_train.shape)
print(y_train.shape)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
