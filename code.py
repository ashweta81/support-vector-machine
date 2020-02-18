# --------------
# Loading the Necessary Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv(path)

plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True)

# Check the correlation between each feature and check for null values
print(df.isnull().sum())

# Print total no of labels also print number of Male and Female labels
print('The total no of labels is', df['label'].count())
print('The total no of males are', df[df['label']=='male'].shape[0])
print('The total no of females are', df[df['label']=='female'].shape[0])
# Label Encode target variable
df['label'] = df['label'].replace({'male': '1', 'female':'0'})
print(df['label'].value_counts())


# Scale all the independent features and split the dataset into training and testing set.
scaler = StandardScaler()
X = df.iloc[:,:-1]
y= df.iloc[:,-1]
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=0)
# Build model with SVC classifier keeping default Linear kernel and calculate accuracy score.
model1 = SVC(kernel='linear')
model1.fit(X_train, y_train)
y_pred = model1.predict(X_test)
acc1 = accuracy_score(y_test,y_pred)
print(acc1)
# Build SVC classifier model with polynomial kernel and calculate accuracy score
model2 = SVC(kernel='poly')
model2.fit(X_train, y_train)
y_pred = model2.predict(X_test)
acc2 = accuracy_score(y_test,y_pred)
print(acc2)

# Build SVM model with rbf kernel.
model3 = SVC(kernel='rbf')
model3.fit(X_train, y_train)
y_pred = model3.predict(X_test)
acc3 = accuracy_score(y_test,y_pred)
print(acc3)

#  Remove Correlated Features.
d1 = df.drop('label', 1).corr().abs()
d2 = d1.where(np.triu(np.ones(d1.shape),k=1).astype(np.bool))
cols_to_drop = [cols for cols in d2.columns if any (d2[cols])>0.95]
print(cols_to_drop)
# Split the newly created data frame into train and test set, scale the features and apply SVM model with rbf kernel to newly created dataframe


# Do Hyperparameter Tuning using GridSearchCV and evaluate the model on test data.





