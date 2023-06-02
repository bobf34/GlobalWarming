import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pdb
from scipy.fft import fft,ifft,fftshift
from getTempSunspotData import getTempSunspotData
import os
def reformatONI(df):
        df = df.set_index('YEAR').stack().reset_index().rename({'YEAR':'Year','level_1': 'month',0:'ONI'}, axis=1)
        df.Year = df.Year[0] + df.index.values/12
        df = df.drop('month',axis=1)
        df = df.loc[np.where(df.ONI>-999)]
        return df

def getENSOdata(useLocal = True, plotData=False, filename='enso_oni.csv'):

    if useLocal and os.path.isfile(filename):
        df = pd.read_csv(filename)
        if plotData:
           df.plot(x='Year')
           plt.show()
    else:
        url = 'https://psl.noaa.gov/enso/mei.ext/table.ext.html'
        dfext = pd.read_csv(url,sep='\s+',skiprows=11,skipfooter=10,engine='python')
        colnames = list(dfext)
        dfext = reformatONI(dfext)
        url = 'https://psl.noaa.gov/enso/mei/data/meiv2.data'
        dfcur = pd.read_csv(url,sep='\s+',skiprows=1,skipfooter=4,names=colnames, engine='python')
        dfcur = reformatONI(dfcur)
        if plotData:
            ax=dfext.plot(x='Year')
            dfcur.plot(ax=ax,x='Year')
            plt.show()
        dfext = dfext.loc[np.where(dfext.Year<dfcur.Year[0])]
        df = pd.concat([dfext,dfcur],ignore_index=True)
        df.to_csv(filename,index=False)

    return df

class ENSO:
    def __init__(self,yearOffset=0):
        self.df = getENSOdata(useLocal = True, plotData = False)
        self.df.Year += yearOffset

    def getEvents(self,threshold=2.0):
            if threshold>0:
               eventLocs = np.append([0],np.where(np.convolve(self.df.ONI,np.ones(2)/2,mode='same')>threshold))
            else:
               eventLocs = np.append([0],np.where(np.convolve(self.df.ONI,np.ones(2)/2,mode='same')<threshold))
            startEventLocs = eventLocs[np.where(np.diff(eventLocs)>18)[0]+1]
            df_events = self.df.loc[startEventLocs].copy().reset_index(drop=True)
            return df_events

    def dropEventDups(self,A,B,yearDiff = 1):
        # removes from rows from A where year is close to row in B
           vStrDups = np.where(np.abs(np.subtract.outer(A.Year.values,B.Year.values))<yearDiff)[0]
           return A.drop(vStrDups).reset_index(drop=True)


if __name__ == "__main__":
    getENSOdata(useLocal = True, plotData = True) 
