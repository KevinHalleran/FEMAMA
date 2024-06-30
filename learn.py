import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import learn

#https://www.webscale.com/engineering-education/multivariate-time-series-using-auto-arima/

class learn:
    def getFileAsDataFrame(filename):
        return pd.read_csv(filename,index_col=False)

    def CleanUpAddresses():
        df=learn.getFileAsDataFrame('MissionAssignments.csv')
        #print('remove dashes')
        df['city']=df['city'].str.replace('-','');

        #print('remove commas')
        df['city']=df['city'].str.replace(',','');

        #print('remove whitespace')
        df['city']=df['city'].str.strip();

        df.to_csv('MissionAssignments-Revised.csv')

    def LinearRegression():
        print('get file')
        df=learn.getFileAsDataFrame('MissionAssignments.csv')
        df.head()
        print(df.head(10))

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
    def ARIMA(df):
        #https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
        
        #convert to dates then to floats from the date
        df['dateRequested']=pd.to_datetime(df['dateRequested'])        
        df['requestedAmount']=pd.to_numeric(df['requestedAmount'])
        df=learn.getTrainSet(df)
        df=df.dropna()
        #print(df['dateRequested'])

        #Convert to ordinal
        df['dateRequested']=df['dateRequested'].apply(lambda x: x.toordinal())

        #create an array
        data = pd.Series(df['requestedAmount']).to_numpy()
        #data = pd.Series(df['dateRequested']).to_numpy()

        # Fit the ARIMA model
        #https://stackoverflow.com/questions/31690134/python-statsmodels-help-using-arima-model-for-time-series
        #https://365datascience.com/tutorials/python-tutorials/arima/
        #https://www.capitalone.com/tech/machine-learning/arima-model-time-series-forecasting/
        #https://www.machinelearningplus.com/time-series/time-series-analysis-python/
        #https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/
        model = ARIMA(data, order=(3, 2, 2))
        model_fit = model.fit()

        # Predict
        predictions = model_fit.forecast(steps=3)[0]
        print(model_fit.summary())
        #print(predictions)
    def getTrainSet(df):
        return df[df['dateRequested']< '10/1/2021']
    
    def getTestSet(df):
        return df[df['dateRequested']>= '10/1/2021']
    
    def plotDF():
        print('load file')
        df=learn.getFileAsDataFrame('MissionAssignments.csv') 
        df=df[['dateRequested','requestedAmount']]

        #convert to dates then to floats from the date
        df['dateRequested']=pd.to_datetime(df['dateRequested'])        
        df['requestedAmount']=pd.to_numeric(df['requestedAmount'])

        df=df.dropna()

        print('get train set')
        df=learn.getTrainSet(df)
        #df['dateRequested'].info()
        print('fix times')
        df['dateRequested']=df['dateRequested'].apply(pd.Timestamp.toordinal)
        df['dateRequested'].plot()
        #print(df['dateRequested'].head(50))
        #print(df['dateRequested'].tail(50))
        
        #df.info()
        acf_original = plot_acf(df['dateRequested'])
        pacf_original = plot_pacf(df['dateRequested'])
        #print('show plot')
        #plt.show()
        
        #print('tune')
        #autoArima=pm.auto_arima(df['dateRequested'],stepwise=False,seasonal=False)
        #autoArima

    def FindD():
        df=learn.getFileAsDataFrame('MissionAssignments.csv') 
        #df_learn=df[['dateRequested','requestedAmount']]
        df_learn=df[['dateRequested']]
        df_learn['dateRequested']=pd.to_datetime(df_learn['dateRequested'])        
        #df_learn['requestedAmount']=pd.to_numeric(df_learn['requestedAmount'])
        df_learn=df_learn.dropna()

        df_learn['dateRequested']=df_learn['dateRequested'].apply(lambda x: x.toordinal())

        result = adfuller(df_learn.dropna())
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
    def autoCorrelate(df):
        plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

        # Import data
        df_learn=df[['requestedAmount']]
        #df_learn['dateRequested']=pd.to_datetime(df_learn['dateRequested'])        
        df_learn['requestedAmount']=pd.to_numeric(df_learn['requestedAmount'])
        df_learn=df_learn.dropna()

        #df_learn['dateRequested']=df_learn['dateRequested'].apply(lambda x: x.toordinal())
        df_learn=df_learn.dropna()

        # Original Series
        
        fig, axes = plt.subplots(3, 2, sharex=True)
        axes[0, 0].plot(df_learn); axes[0, 0].set_title('Original Series')
        plot_acf(df_learn, ax=axes[0, 1])

        # 1st Differencing
        axes[1, 0].plot(df_learn.diff()); axes[1, 0].set_title('1st Order Differencing')
        plot_acf(df_learn.diff().dropna(), ax=axes[1, 1])

        # 2nd Differencing
        axes[2, 0].plot(df_learn.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
        plot_acf(df_learn.diff().diff().dropna(), ax=axes[2, 1])

        plt.show()
    def visualizeSeries(df):
        df_learn=df[['dateRequested','requestedAmount']]

        df_learn['dateRequested']=pd.to_datetime(df_learn['dateRequested'])        
        df_learn['requestedAmount']=pd.to_numeric(df_learn['requestedAmount'])
        df_learn=df_learn.dropna()

        #https://www.geeksforgeeks.org/how-to-plot-a-pandas-dataframe-with-matplotlib/
        # Draw Plot
        def plot_df(df_learn, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
            plt.figure(figsize=(16,5), dpi=dpi)
            plt.plot(x, y, color='tab:red')
            plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
            plt.show()

        plot_df(df_learn, x=df_learn['dateRequested'], y=df_learn['requestedAmount'],title='')    
    
    def byEvent(df, eventId):
        df=df[df['disasterNumber']==eventId]
        learn.ARIMA(df)    
        #learn.autoCorrelate(df)
        #learn.visualizeSeries(df)



print('get file')
df=learn.getFileAsDataFrame('MissionAssignments.csv') 
eventId=4339
learn.byEvent(df, eventId)
#learn.plotDF()
#learn.LinearRegression()
#learn.ARIMA()
#learn.FindD()
#learn.autoCorrelate()
#learn.visualizeSeries()
print('complete')
