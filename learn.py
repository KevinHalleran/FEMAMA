import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
#from tensorflow.python.keras.models import Sequential
#from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.layers import LSTMV1
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
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            aVal = dataset[i:(i+look_back), 0]
            dataX.append(aVal)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)
    def LSTMRecursion(df, trainId, testId):        
        #https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
        tf.random.set_seed(7)
        df=df[['dateRequested','requestedAmount','disasterNumber']]
        df['dateRequested']=pd.to_datetime(df['dateRequested'])        
        df['requestedAmount']=pd.to_numeric(df['requestedAmount'])
        df=df.dropna()
        df['dateRequested']=df['dateRequested'].apply(lambda x: x.toordinal())
        df=df.dropna()

        df_train=df[df['disasterNumber']==trainId]
        df_test=df[df['disasterNumber']==testId]

        df_train=df_train[['dateRequested','requestedAmount']]
        df_test=df_test[['dateRequested','requestedAmount']]

        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        #https://stackoverflow.com/questions/62178888/can-someone-explain-to-me-how-minmaxscaler-works
        scaler = MinMaxScaler(feature_range=(0, 1))
        df_train=scaler.fit_transform(df_train)
        df_test=scaler.fit_transform(df_test)

        print('reshape into X=t and Y=t+1')
        look_back = 1
        trainX, trainY = learn.create_dataset(df_train, look_back)
        testX, testY = learn.create_dataset(df_test, look_back)

        #https://www.w3schools.com/python/numpy/numpy_array_reshape.asp
        #https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
        print('reshape input to be [samples, time steps, features]')
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        print("trainX.shape[0]:" + str(trainX.shape[0]))
        print("trainX.shape[1]:" + str(trainX.shape[1]))

        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        print("testX.shape[0]:" + str(testX.shape[0]))
        print("testX.shape[1]:" + str(testX.shape[1]))

        print('create and fit the LSTM network')
        #https://keras.io/guides/sequential_model/
        #https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
        model = keras.Sequential()
        model.add(keras.layers.LSTM(4, input_shape=(1, look_back)))
        model.add(keras.layers.Dense(1))
        model.summary()

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

        #https://keras.io/guides/training_with_built_in_methods/
        #https://www.tensorflow.org/api_docs/python/tf/keras/Model
        model.evaluate(testX, testY)

        #This might save me time in the future, really only run it once and save the results since the data is not changing.
        #https://keras.io/guides/serialization_and_saving/
        print('make predictions')
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        
        print('invert predictions')
        #thank you co-pilot
        #https://stackoverflow.com/questions/45847006/valueerror-operands-could-not-be-broadcast-together-with-shapes-inverse-trans
        #https://datascience.stackexchange.com/questions/22488/value-error-operands-could-not-be-broadcast-together-with-shapes-lstm
        #https://stackoverflow.com/questions/57216718/how-to-inverse-transform-the-predicted-values-in-a-multivariate-time-series-lstm
        #https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

        testPredict = testPredict.reshape((testX.shape[0], testX.shape[2]))
        # invert scaling for forecast
        inv_yhat = np.concatenate((testPredict, testX[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y, testX[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

        # trainPredict = trainPredict.reshape(testX.shape[0],testX.shape[2])

        # trainPredict=np.concatenate(trainPredict,testX[:,1:],axis=1)
        # print(trainPredict.shape)
        # trainPredict = scaler.inverse_transform(trainPredict) #here
        # trainY = scaler.inverse_transform([trainY])
        # testPredict = scaler.inverse_transform(testPredict)
        # testY = scaler.inverse_transform([testY])
        
        # print('calculate root mean squared error')
        # trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        # print('Test Score: %.2f RMSE' % (testScore))

        # print('shift train predictions for plotting')
        trainPredictPlot = np.empty_like(df_train)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        
        # print('shift test predictions for plotting')
        testPredictPlot = np.empty_like(df_test)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(look_back*2)+1:len(df)-1, :] = testPredict

        print('plot baseline and predictions')
        plt.plot(scaler.inverse_transform(df))
        plt.plot(trainPredictPlot)
        plt.plot(testPredictPlot)
        plt.show()

print('get file')
df=learn.getFileAsDataFrame('MissionAssignments.csv') 
eventId=4339 #Maria
testId=4332 #Harvey
#learn.byEvent(df, eventId)
learn.LSTMRecursion(df,eventId,testId)
#learn.plotDF()
#learn.LinearRegression()
#learn.ARIMA()
#learn.FindD()
#learn.autoCorrelate()
#learn.visualizeSeries()
print('complete')
