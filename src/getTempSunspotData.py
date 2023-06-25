import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import logging
import os
    
def getTempSunspotData(useLocal = True, plotData=False, temp_filename = 'globalTempAnomalies.csv', ss_filename = 'sunspots.csv'):
    ########### SUNSPOTS ########################
    #Source: WDC-SILSO, Royal Observatory of Belgium, Brussels
    #website: https://www.sidc.be/silso/datafiles
    if useLocal and os.path.isfile(ss_filename):
        df_ss=pd.read_csv(ss_filename)
    else:
        url = 'https://www.sidc.be/silso/INFO/snmtotcsv.php'
        logging.debug('Gathering sunspot data from '+url+' ...')
        df_ss=pd.read_csv(url, encoding="ISO-8859-1",header=None, delimiter='\;', engine='python')
        df_ss.drop(columns=[0,1,4,5,6],inplace=True)
        df_ss.rename(columns={df_ss.columns[0]: 'Year',df_ss.columns[1]:'Sunspots'}, inplace=True)
        logging.debug('Saving datat to: '+ss_filename)
        df_ss.to_csv(ss_filename,index=False)
    
    
    ########### GLOBAL TEMPERATURE ########################
    #Source: NOAA
    #website: https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies
    if useLocal and os.path.isfile(temp_filename):
        df_temp = pd.read_csv(temp_filename)
    else:
        url = 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/all/12/1850-2023.csv'
        logging.debug('Gathering Temperature data from '+url+' ...')
        df_temp=pd.read_csv(url, encoding="ISO-8859-1",header=4)
        pdb.set_trace()
        df_temp.rename(columns={"Anomaly": "Temperature"},inplace=True)
        #convert to decimal year e.e. 185001, 185002 to 1850.042, 1850.125
        x = np.modf(df_temp.Year/100)
        df_temp.Year = x[1] + (x[0]*100-1)/12+1/24
        #drop data prior to 1880
        df_temp = df_temp[df_temp.Year>1880]
        df_temp.index = np.arange(len(df_temp))
        logging.debug('Saving datat to: '+temp_filename)
        df_temp.to_csv(temp_filename,index=False)
    
    if plotData:
        fig, (ax_temp, ax_ss) = plt.subplots(2, 1, sharex=False)
        ax_ss.plot(df_ss.Year,df_ss.Sunspots)
        ax_ss.grid()
        ax_ss.set_title('Sunspots')
    
        ax_temp.plot(df_temp.Year,df_temp.Temperature,'r')
        ax_temp.grid()
        ax_temp.set_title('Global Temperature Anomalies')
        plt.show()
    return [df_temp,df_ss] 

if __name__ == "__main__":
    getTempSunspotData(useLocal = True, plotData=True)
