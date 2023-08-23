import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt
import logging
import os

def getHC5TempData(useLocal = False, plotData=False,temp_filename = 'hc5TempAnomalies.csv'):
    ########### GLOBAL TEMPERATURE ########################
    #Source: Met Office
    #website: https://www.metoffice.gov.uk/hadobs/hadcrut5
    '''
     HadCRUT5 is subject to Crown copyright protection and is provided under the Open Government License v3. When publishing work using the data, please use the following citation:
     Morice, C.P., J.J. Kennedy, N.A. Rayner, J.P. Winn, E. Hogan, R.E. Killick, R.J.H. Dunn, T.J. Osborn, P.D. Jones and I.R. Simpson (in press) An updated assessment of near-surface temperature change from 1850: the HadCRUT5 dataset. Journal of Geophysical Research (Atmospheres) doi:10.1029/2019JD032361

     A term of the licence is that users must include the following acknowledgement when the data are used:

         HadCRUT.[version number] data were obtained from http://www.metoffice.gov.uk/hadobs/hadcrut5 on [date downloaded] and are © British Crown Copyright, Met Office [year of first publication], provided under an Open Government License, http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/ 
    '''
    if useLocal and os.path.isfile(temp_filename):
        df_temp = pd.read_csv(temp_filename)
    else:
        url = 'https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/diagnostics/HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv'
        logging.debug('Gathering Temperature data from '+url+' ...')
        df_temp=pd.read_csv(url)
        df_temp.rename(columns={"Anomaly (deg C)": "Temperature","Time":"Year"},inplace=True)
        df_temp[['Year','Month']] = df_temp['Year'].str.split('-',expand=True)
        df_temp.Year = df_temp.Year.astype(float)
        df_temp.Month = df_temp.Month.astype(float)
        df_temp.Year += df_temp.Month/12 - 1/24

        df_temp.drop(columns=['Month','Lower confidence limit (2.5%)','Upper confidence limit (97.5%)'],inplace=True)
        #df_temp.index = np.arange(len(df_temp))
        logging.debug('Saving datat to: '+temp_filename)

        if plotData:
            df_temp.plot(x='Year')

        df_temp.to_csv(temp_filename,index=False)
    return df_temp
 
def getNOAATempData(useLocal = False, temp_filename = 'globalTempAnomalies.csv'):
    ########### GLOBAL TEMPERATURE ########################
    #Source: NOAA
    #website: https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies
    if useLocal and os.path.isfile(temp_filename):
        df_temp = pd.read_csv(temp_filename)
    else:
        url = 'https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/all/12/1850-2023.csv'
        logging.debug('Gathering Temperature data from '+url+' ...')
        df_temp=pd.read_csv(url, encoding="ISO-8859-1",header=4)
        df_temp.rename(columns={"Anomaly": "Temperature"},inplace=True)
        #convert to decimal year e.e. 185001, 185002 to 1850.042, 1850.125
        x = np.modf(df_temp.Year/100)
        df_temp.Year = x[1] + (x[0]*100-1)/12+1/24
        #drop data prior to 1880
        #df_temp = df_temp[df_temp.Year>1880]
        df_temp.index = np.arange(len(df_temp))
        logging.debug('Saving datat to: '+temp_filename)
        df_temp.to_csv(temp_filename,index=False)
    return df_temp

def getSunspotData(useLocal = True, ss_filename = 'sunspots.csv'):
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
    return df_ss

def getTempSunspotData(useLocal = True, plotData=False, 
        tempSrc='HC5', temp_filename = 'globalTempAnomalies.csv', 
        ss_filename = 'sunspots.csv'):

    df_ss = getSunspotData(useLocal = useLocal, ss_filename = ss_filename)
    if tempSrc=='HC5':
        df_temp = getHC5TempData(useLocal = useLocal, temp_filename = 'HC5'+temp_filename)
    elif tempSrc=='NOAA':
        df_temp = getNOAATempData(useLocal = useLocal, temp_filename = 'NOAA'+temp_filename)
    else:
        df_temp = getHC5TempData(useLocal = useLocal, temp_filename = temp_filename)

    if plotData:
        fig, (ax_temp, ax_ss) = plt.subplots(2, 1, sharex=False)
        ax_ss.plot(df_ss.Year,df_ss.Sunspots,label='WSN')
        ax_ss.grid()
        ax_ss.legend()
        ax_ss.set_title('Sunspots')

        ax_temp.plot(df_temp.Year,df_temp.Temperature,'r')
        ax_temp.grid()
        ax_temp.set_title(tempSrc+' Global Temperature Anomalies')
        plt.show()
    return [df_temp,df_ss]

if __name__ == "__main__":
    getTempSunspotData(useLocal = True, tempSrc= 'HC5', plotData=True)

