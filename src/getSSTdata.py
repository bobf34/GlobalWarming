import netCDF4 as nc
from matplotlib import pyplot as plt, patches
from netCDF4 import num2date
import pandas as pd
import numpy as np
import pdb
import requests
import os
import sys

def downloadSST(filename = 'sst.mnmean.nc'):
   '''
   Boyin Huang, Peter W. Thorne, Viva F. Banzon, Tim Boyer, Gennady Chepurin, Jay H. Lawrimore, 
   Matthew J. Menne, Thomas M. Smith, Russell S. Vose, and Huai-Min Zhang (2017): 
   NOAA Extended Reconstructed Sea Surface Temperature (ERSST), Version 5. [indicate subset used]. 
   NOAA National Centers for Environmental Information. 
   doi:10.7289/V5T72FNM. Obtain at NOAA/ESRL/PSD at their website https://www.esrl.noaa.gov/psd/ [access date].
   '''
   #https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
   
   url = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/"+filename

   print("Downloading "+ filename +" ...")
   sys.stdout.flush()
   r = requests.get(url)
   with open(filename, 'wb') as f:
       #opening the file in write mode
       f.write(r.content)
       f.close()
   print("Done")


def convertSSTNetCDF(filename = 'sst.mnmean.nc',movingAverageMonths=0):
   #download the file
   if not os.path.isfile(filename):
        downloadSST(filename)

   ds = nc.Dataset('sst.mnmean.nc')
   #print(df.dimensions)
   #print(df.variables)


   # Open the netCDF file
   with ds as dataset:
      # Access variables and their attributes
      sst_variable = dataset.variables['sst']
      time_variable = dataset.variables['time']

      # Convert time values to datetime objects
      times = num2date(time_variable[:], units=time_variable.units,only_use_cftime_datetimes=False)

      #average over lat lons
      global_average= np.nanmean(sst_variable[:,:,:],axis=(1,2))
      year = pd.DatetimeIndex(times).year + pd.DatetimeIndex(times).month/12-1/24

   # Create DataFrame with time, latitude, longitude, expver, t2m, and tp values
   df = pd.DataFrame({'Year':year,'Temperature':global_average})
   if movingAverageMonths > 0:
      df['MovingAverage']=df.Temperature.rolling(movingAverageMonths).mean()
   return(df)

def getSSTdata(filename = 'sst.mnmean.nc', plotData=False, movingAverageMonths=0):
   df = convertSSTNetCDF(filename = filename, movingAverageMonths=movingAverageMonths)
   if plotData: 
      plt.plot(df.Year,df.Temperature,'0.7',label="Sea Surface Temp")
      if movingAverageMonths:
         plt.plot(df.Year,df.MovingAverage,'r',label="{} Month MA".format(movingAverageMonths))
      plt.xlabel="Year"
      plt.ylabel="°C"
      plt.grid()
      plt.legend()
   return(df)

if __name__ == "__main__":
    getSSTdata(plotData=True,movingAverageMonths=36)
    plt.show()
