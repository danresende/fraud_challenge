# Loading libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Getting data
data_path = os.path.join(os.getcwd(), 'Raw data/Fraud Instance  Raw Data.xlsx - Raw.csv')

data = pd.read_csv(data_path, index_col=0)

# Cleaning data
data['Claim Amount'] = data['Claim Amount'].replace('[\$,]', '', regex=True).astype(int)

# Create dummy variables
data = pd.get_dummies(data, columns=['Marital Status', 'Accomodation Type'],
        drop_first=True)

# Split train and test data
y = data['Fraud Instance'].values
X = data.drop(['Fraud Instance'], axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Standardize data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Create model
reg = LogisticRegression()
reg = reg.fit(X_train, y_train)

# Predict test data
prediction = reg.predict(X_test)

# Prediction accuracy
acc = reg.score(X_test, y_test)
print(acc)
