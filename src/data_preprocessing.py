# +
import pandas as pd
import numpy as np

data = pd.read_csv('creditcard.csv')
data.describe()
# -
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]
outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)
print('Fraud Cases: {}'.format(len(data[data['Class'] == 1])))
print('Valid Transactions: {}'.format(len(data[data['Class'] == 0])))

print("Amount details of the fraudulent transaction")
fraud.Amount.describe()

print("Details of valid transaction")
valid.Amount.describe()


