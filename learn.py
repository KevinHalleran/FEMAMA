import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv('MissionAssignments.csv')
df.head()
print(df.head(10))

#print('remove dashes')
#df['city']=df['city'].str.replace('-','');

#print('remove commas')
#df['city']=df['city'].str.replace(',','');

#print('remove whitespace')
#df['city']=df['city'].str.strip();

#df.to_csv('MissionAssignments-Revised.csv')

print('shrink original dataset')
#df_learn=df[['maid','disasterNumber','declarationType','disasterDescription','dateRequested','requestedAmount','obligationAmount']]
df_learn=df[['dateRequested','requestedAmount']]

print('convert data types')
df_learn['dateRequested']=pd.to_datetime(df_learn['dateRequested'])
df_learn['requestedAmount']=pd.to_numeric(df_learn['requestedAmount'])
print(df_learn.head(10))

print('Define training and testing sets')
df_train=df[df_learn['dateRequested']< '10/1/2021']
df_test=df[df_learn['dateRequested']>= '10/1/2021']

#https://stackoverflow.com/questions/24588437/convert-date-to-float-for-linear-regression-on-pandas-data-frame
print('make dates floats')
df_train['dateRequested']=pd.to_datetime(df_train['dateRequested'])
df_train['dateRequested']=df_train['dateRequested'].apply(lambda x: x.toordinal())

df_test['dateRequested']=pd.to_datetime(df_test['dateRequested'])
df_test['dateRequested']=df_test['dateRequested'].apply(lambda x: x.toordinal())

#https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
print('convert to numpy')
np_train_x=df_train['dateRequested'].values[:,np.newaxis]
np_train_y=df_train['requestedAmount'].values[:,np.newaxis]

np_test_x=df_test['dateRequested'].values[:,np.newaxis]
np_test_y=df_test['requestedAmount'].values[:,np.newaxis]

print('create model')
reg = linear_model.LinearRegression()

print('train model')
reg.fit(np_train_x,np_train_y)

print('test model')
result=reg.predict(np_test_x)

# The coefficients
print("Coefficients: \n", reg.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(np_test_y, result))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(np_test_y, result))

# Plot outputs
plt.scatter(np_test_x, np_test_y, color="black")
plt.plot(np_test_x, result, color="blue", linewidth=3)
plt.legend(loc="upper left")

plt.xticks(())
plt.yticks(())

plt.show()

print('complete')