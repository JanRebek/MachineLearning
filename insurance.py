# import libraries
import matplotlib as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

# import data
df = pd.read_csv('train.csv')

# visualize data

df.Response.value_counts().plot(kind='pie', labels=['No', 'Yes'], title='Interested', startangle=270, autopct='%1.1f%%')

df.Vehicle_Age.value_counts().plot(kind='pie', labels=['> 2 Years', '1-2 Years', '< 1 Year'], title='Vehicle Age', startangle=150, autopct='%1.1f%%')
plt.show()

sns.distplot(df['Age'], kde=False, color='red')
plt.show()

sns.barplot(x="Vehicle_Age", y="Annual_Premium", data=df,
                 color="salmon", saturation=.5)
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

#Training the Logistic Regression model on the training set
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)

# predicting the Test set results
y_pred = model.predict(X_test)

#define new instance
Xnew = [[ 1, 1, 0, 0, 35, 1, 1, 15000, 80 ]] #male, vehicle was damaged, more than two years old vehicle, person is 35 years old, has driving license, was previously insure, annual premium 15000, 80 vintage

# Predicting a new result
y_new = model.predict(Xnew)

# Making the Confusion Matrix and accuracy score
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

print ('cm:')
print(cm)
print(f'accuracy_score: {accuracy_score(y_test, y_pred)}')












