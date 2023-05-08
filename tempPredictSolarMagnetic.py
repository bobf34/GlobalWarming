import pandas as pd
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from getTempSunspotData import getTempSunspotData
from getSynopticData import getSynopticData,cr,yr
    

def getDataInFitRegion(df_tempMA,df_pred):
    # returns the overlapping temperature and prediction data
    firstFitYear = df_pred.Year[0]  #temp starts before pred
    lastFitYear = df_tempMA.Year.iloc[-1] #temp ends before pred

    idxMA =np.where((df_tempMA.Year>=firstFitYear)&(df_tempMA.Year <=lastFitYear))[0]
    tma = df_tempMA.Temperature[idxMA].values
    year = df_tempMA.Year[idxMA].values

    #The temperature and prediction data are not sampled at the same rate.  Use the predictions 
    #with the closest time stamps to the temperature data
    idxPred =[(df_pred['Year']-yr).abs().argsort()[:-1][0] for yr in year]
    pred = df_pred.Temperature[idxPred].values

    return [tma,pred,year]

def fitData(df_tempMA,df_pred):
     # find gain and offset that minimizes error
    [tma,pred,year] = getDataInFitRegion(df_tempMA,df_pred)

    X =  np.vstack([pred,np.ones(len(pred))]).T
    # perform LMS fit
    [gm,c] = np.linalg.lstsq(X, tma, rcond=None)[0]
    err = (tma- np.dot(X,[gm,c])) #err is computed over the fit interval

    fitParms = {'gain':gm, 'offset':c,'rmserr':np.std(err)}
    return fitParms


############  BEGIN MAIN PROGRAM ####################
# Get the temperature data
[df_temp,df_ss] = getTempSunspotData(useLocal = True, plotData=False)

#Create a 3yr moving average of the temperature data for plotting and prediction error purposes
lenMA = 3
tempMA = np.convolve(df_temp.Temperature,np.ones(lenMA*12)/(lenMA*12),mode='valid')
t_tempMA = np.arange(len(tempMA))/12+df_temp.Year[0]+lenMA/2
df_tempMA = pd.DataFrame( {'Year':t_tempMA,'Temperature':tempMA})

# Get the WSO synoptic data
df_syn = getSynopticData()

#Get the latitude column lables
lats = list(df_syn)[3:]

# negate the data in the souther latitudes and sum for row (time step)
southLats = lats[int(len(lats)/2):]
df = df_syn[lats].copy()
df[southLats] *= -1

#For each time, sum over the lower latitudes and compute the absolute value
ignoreLats = 10  # ignore the N highest and lowest latitudes
sumLats = lats[ignoreLats:-ignoreLats]
m = np.abs(df[sumLats].sum(axis=1))

# apply an 11-year moving average to the sum
L = int(11*72*yr/cr)  #length of moving average (11 years) * longitudes * years/Carrington cycle
w = np.ones(L)/L
predict = -np.convolve(m,w,mode='valid')
year= df_syn.Year.values[L-1:]+2.8

#create a dataframe and fit the prediction to the moving-average temperature (scale/offset)
df_pred =  pd.DataFrame({'Year':year, 'Temperature':predict})
fitParms = fitData(df_tempMA,df_pred)
df_pred.Temperature = df_pred.Temperature * fitParms['gain'] + fitParms['offset']
    
plt.title('Temperature Anomalies ')
plt.plot(df_temp.Year,df_temp.Temperature,'.8',label='NOAA Global Temp Anomaly')
plt.plot(df_tempMA.Year,df_tempMA.Temperature,'r')
plt.plot(df_tempMA.Year,df_tempMA.Temperature,'r',label='Temp 3 Yr Moving Average')
errStr = ': RMS Error:'+'{:.4f}'.format(fitParms['rmserr'])+'°C'
plt.plot(df_pred.Year,df_pred.Temperature,'b',label='Prediction from Solar Magnetic Field'+errStr)
plt.ylabel('°C')
plt.xlabel('Year')
plt.xlim(1980,2030)
plt.ylim(0,1.5)
plt.legend()
plt.grid()

plt.show()

