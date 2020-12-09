import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# import data
df = pd.read_csv('train.csv')

# make X and y

X_columns = ['Gender', 'Age', 'Driving_License', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Vintage']
X = df.loc[:, X_columns]
X_original = X
y = df.iloc[:, 11:12]
y = y.values

# Splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# get categories using ColumnTransformer and OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['Gender', 'Vehicle_Damage', 'Vehicle_Age'])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train)) # First column is Gender, second is Vehicle_Damage, third and fourth are about Vehicle_Age
X_test = np.array(ct.fit_transform(X_test)) # First column is Gender, second is Vehicle_Damage, third and fourth are about Vehicle_Age


# feature scaling, min-max scaler (0-1 range)
scaler_x = MinMaxScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

# build model, use relu & sigmoid as the output activation function
model = Sequential()
model.add(Dense(12, input_dim=9, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=32,  verbose=1, validation_data=(X_test,y_test))

# visualize training
print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# predict test
y_pred = model.predict(X_test)

# predict X_test
y_pred = model.predict_classes(X_test)

# Making the Confusion Matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print ('cm:')
print(cm)
print(f'accuracy_score: {accuracy_score(y_test, y_pred)}')

#save model to my_model directory
model.save('my_model')