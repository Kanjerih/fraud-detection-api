# +
import pandas as pd
import numpy as np

data = pd.read_csv('creditcard.csv')
data.describe()
# +
import seaborn as sns

sns.histplot(data.Time, kde=True)


# +
import seaborn as sns

sns.histplot(data.Class, kde=True)
# -

data.Class

data.Amount

sns.histplot(data.Amount, kde =True
            )

sns.histplot(data.V14, kde =True
            )


