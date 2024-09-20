import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from tensorflow.python.keras.engine import data_adapter
import tensorflow.python.keras as tf_keras
from keras import __version__
tf_keras.__version__ = __version__


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

# Drop rows with missing values
df.dropna(inplace=True)

# Define the columns
categorical_columns = ['Scholarship type', 'Regular artistic or sports activity']
numerical_columns = ['Student Age', 'Additional work', 'Total salary if available',
                     'Weekly study hours', 'Attendance to classes',
                     'Preparation to midterm exams 1',
                     'Cumulative grade point average in the last semester (/4.00)']

# Separate features and target
X = df[numerical_columns + categorical_columns]
y = df['GRADE']


def scale_dataset(dataframe, oversample=False):
    X = dataframe[numerical_columns + categorical_columns]
    y = dataframe['GRADE']

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    # Preprocess data: scale numerical features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    X_transformed = preprocessor.fit_transform(X)

    return X_transformed, y


# Split data into training, validation, and test sets using pandas directly
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.6 * len(df))
validation_size = int(0.8 * len(df))

train = df_shuffled.iloc[:train_size]
validation = df_shuffled.iloc[train_size:validation_size]
test = df_shuffled.iloc[validation_size:]

# Scale datasets
X_train, y_train = scale_dataset(train, True)
X_valid, y_valid = scale_dataset(validation, False)
X_test, y_test = scale_dataset(test, False)

# Convert target variable to categorical
y_train = to_categorical(y_train, num_classes=8)
y_valid = to_categorical(y_valid, num_classes=8)
y_test = to_categorical(y_test, num_classes=8)

# Model Setup
input_shape = X_train.shape[1]  # Number of features after encoding

model = Sequential([
    Dense(64, activation='relu', input_shape=(14,)),  # Adjust input_shape to match number of features
    Dropout(0.01),
    Dense(32, activation='relu'),  # Hidden layer with 32 units
    Dropout(0.001),
    Dense(16, activation='relu'),  # Hidden layer with 16 units
    Dense(8, activation='softmax')  # Output layer with 5 units (for 5 classes)
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

history_df = pd.DataFrame(history.history)
history_df['epochs'] = history_df.index + 1  # Add epochs as a column
history_df.to_csv('history.csv', index=False)

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

model.save('my_model.h5')
