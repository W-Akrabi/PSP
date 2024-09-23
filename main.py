import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

# Load the dataset
df = pd.read_csv("data_col.csv")

# Filter the DataFrame
df = df[['Student Age', 'Scholarship type', 'Additional work',
         'Regular artistic or sports activity', 'Total salary if available',
         'Weekly study hours', 'Attendance to classes',
         'Preparation to midterm exams 1',
         'Cumulative grade point average in the last semester (/4.00)',
         'GRADE']]

# Drop rows with missing values
df.dropna(inplace=True)

# Define columns
categorical_columns = ['Scholarship type', 'Regular artistic or sports activity', 'Attendance to classes']
numerical_columns = ['Student Age', 'Additional work', 'Total salary if available',
                     'Weekly study hours', 'Preparation to midterm exams 1',
                     'Cumulative grade point average in the last semester (/4.00)']

# Separate features and target
X = df[numerical_columns + categorical_columns]
y = df['GRADE']

def scale_dataset(train_dataframe, test_dataframe, oversample=False):
    X_train = train_dataframe[numerical_columns + categorical_columns]
    y_train = train_dataframe['GRADE']

    if oversample:
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X_train, y_train)

    # Preprocess data: scale numerical features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(test_dataframe[numerical_columns + categorical_columns])

    return X_train_transformed, y_train, X_test_transformed

# Split data into training, validation, and test sets
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_size = int(0.6 * len(df))
validation_size = int(0.8 * len(df))

train = df_shuffled.iloc[:train_size]
validation = df_shuffled.iloc[train_size:validation_size]
test = df_shuffled.iloc[validation_size:]

# Scale datasets
X_train, y_train, X_valid = scale_dataset(train, validation, True)
X_test, y_test = scale_dataset(train, test, False)[:2]  # Get both X_test and y_test

# Convert target variable to categorical
y_train = to_categorical(y_train, num_classes=8)
y_valid = to_categorical(validation['GRADE'], num_classes=8)  # Convert validation target to categorical
y_test = to_categorical(y_test, num_classes=8)

# Check the shape of X_train to ensure correct input size
print("X_train shape:", X_train.shape)  # Check the number of features

# Model Setup
input_shape = X_train.shape[1]  # Number of features after encoding

# Model Setup
model = Sequential([
    Input(shape=(input_shape,)),  # Input layer
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dense(8, activation='softmax')  # Output layer for 8 classes
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping and Model Checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stopping],
    verbose=1
)

# Save the model
model.save('my_model.h5')

# Evaluate the model on test data
model.load_weights('my_model.h5')  # Load the best model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save training history
history_df = pd.DataFrame(history.history)
history_df['epochs'] = history_df.index + 1  # Add epochs as a column
history_df.to_csv('history.csv', index=False)

# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
