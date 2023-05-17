import pandas as pd
import numpy as np
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
import logging
import os
    
cr = 27.2753 #days Carrington Rotation
yr = 365.25 #days

def getTimeIdx(startCycle,endCycle):
    # return array with decimal year with 72 times  per cycle (360 to 5 deg in 5deg increments)
    # NOTE:  Does not account for variations in cycle length so time stames will not always match
    # dates associated with time stamps on http://wso.stanford.edu/synopticl.html
    cr_start= 1642 
    cr_startDate = 1976.402464066 # 1976:05:27 
    numPts = (endCycle-startCycle+1)*72
    step = cr/yr/72
    cr_diff_years = (startCycle - cr_start)*cr/yr
    year = cr_startDate+ np.arange(numPts)*step+cr_diff_years
    return year

def getCycle(cycle):
    # read a cycle file from the Wilcox Solar Observatory

    url = 'http://wso.stanford.edu/synoptic/WSO.'+str(cycle)+'.F.txt'
    #e.g. http://wso.stanford.edu/synoptic/WSO.1642.F.txt

    try: #try to use data with interolated missing data
       #the data isn't fixed width, and on a few values, there's no space between numbers
       #so split on minus sign followed by a number and then get rid of the resulting N/A columns
       df = pd.read_csv(url, sep='\s*(-?[0-9\.]+)', skiprows=2, engine='python', header=None).dropna(thresh=1,axis=1)
       df.drop(df.columns[2], axis=1, inplace=True)  # drop ':' column  will fail on this line if file is empty
    except:  #fall back to regular data, e.g. for CT2208
       url = 'http://wso.stanford.edu/synoptic/WSO.'+str(cycle)+'.txt'
       df = pd.read_csv(url, sep='\s*(-?[0-9\.]+)', skiprows=2, engine='python', header=None).dropna(thresh=1,axis=1)
       df.drop(df.columns[2], axis=1, inplace=True)

    #combine 4 rows into one
    df = pd.concat([df[0::4].reset_index(drop=True), 
                 df[1::4].reset_index(drop=True), 
                 df[2::4].reset_index(drop=True), 
                 df[3::4].reset_index(drop=True)],
                 #ignore_index=True,axis=1)
                 axis=1)
    df.dropna(axis=1,inplace=True)  #drop columns that are all na's
    #put the name cycle name back together
    df.iloc[:,0]=df.iloc[:,0]+df.iloc[:,1].astype(int).astype(str)
    #drop last row as it's a duplicate of the first row in the next cycle
    df = df.iloc[:-1]

    #rename the columns 
    lats = [int(np.arcsin(-14.5/15 + i/15)*180/np.pi) for i in range(30)]
    #colNames = ['Cycle','Longitude'] +lats.astype(int).tolist()
    colNames = ['Cycle', 'Number', 'Longitude', 
                'p75', 'p64', 'p56', 'p50', 'p44', 'p39', 'p34', 'p30', 'p25', 'p21', 'p17', 'p13', 'p9', 'p5', 'p1', 
                'm1', 'm5', 'm9', 'm13', 'm17', 'm21', 'm25', 'm29', 'm34', 'm39', 'm44', 'm50', 'm56', 'm64', 'm75']
    if len(df.columns) != len(colNames):
        pdb.set_trace()
    df.columns = colNames
    df.drop('Number',axis=1,inplace=True)  #don't need both Cycle name and number
    df.Longitude = df.Longitude.astype(int)

    return df

def getSynopticData(useLocal = True, start=1642,end=2258, filename = 'synoptic.csv'):
    # read the database from file, or create csv file from WSO text files
    if useLocal and os.path.isfile(filename):
        df= pd.read_csv(filename)
        cycles = ['CT'+str(i) for i in range(start,end+1)]
        df = df.loc[df['Cycle'].isin(cycles)]

    else:
       df = None
       for r in range(start,end+1):
           print(r)
           if df is None:
              df = getCycle(r)
           else:
              df = pd.concat([df,getCycle(r)])
       #add a time index using decimal year
       year = getTimeIdx(start,end)
       cols = list(df)
       df['Year'] = year
       
       #reorder so Year is first column
       cols = ['Year'] + cols
       df = df[cols]
       #save the file
       df.to_csv(filename,index=False)
       
    return df

if __name__ == "__main__":
    start= 1642 # 1976:05:27 1976.402464066
    end = 2258 # 2022:05:28
    cycle = 'CT2258' #'CT2130'
    showSynoptic = True
    showButterfly = True
    showMagFields = True
    
    
    df = getSynopticData(useLocal = True, start=start, end=end, filename='synoptic.csv')
    dCols = list(df)[3:]

    cycles = np.unique(df.Cycle)  #get list of cycles in dataframe
    #pick the first cycle if cycle isn't in df
    if cycle not in cycles:
        cycle = cycles[0]
    
    lats = [int(np.arcsin(-14.5/15 + i/15)*180/np.pi) for i in range(30)]

    if showSynoptic:  #for a single Carrington Rotation
        # format data for contour plot
       cycleDate = int(1000*df[(df["Cycle"]==cycle) & (df["Longitude"]==180)].Year.values[0])/1000
       df_new = df[df['Cycle'] == cycle]
       longs = df_new.Longitude.values
       Z =  df_new[dCols].values.T
       #levels = np.linspace(Z.min(), Z.max(), 7)
       levels = [-2000,-1000,-500,-100,0,100,500,1000,2000]  #matches Wilcox Solar Observator synoptic charts
       lats.reverse()

       #plt.style.use('_mpl-gallery-nogrid')
       fig, ax = plt.subplots(figsize=(8, 6))

       ax.contourf(longs, lats, Z, levels=levels,cmap=mpl.colormaps['Blues'].reversed())
       ax.contour(longs, lats, Z,levels=levels,colors='k',linewidths=0.5,negative_linestyles='dashed')
       ax.set_title("Cycle: "+cycle+' (center = '+str(cycleDate)+')')
       ax.set_ylabel('Latitude')
       ax.set_xlabel('Longitude')
       xt = [i*30 for i in range(13)]
       ax.set_xticks([i*90 for i in range(5)],minor=False)
       ax.set_xticks([i*30 for i in range(13)],minor=True)
       ax.set_yticks([-60+i*30 for i in range(5)],minor=False)
       #ax.set_yticks([-55, 55, 55], minor=True)
       ax.grid()
    
    if showButterfly:
       df2 = df.copy()
       for lat in dCols:
           df2[lat] = np.convolve(df2[lat],np.ones(72)/72,mode='same')
       Z =  df2[dCols].values.T
       fig2, ax2 = plt.subplots(figsize=(8, 6))
       ax2.pcolormesh(df.Year, lats, Z, vmin=-500, vmax=500,cmap=mpl.colormaps['seismic'].reversed())
       ax2.set_title("\"Butterfly Diagram\" -- Longitudinally Averaged Magnetic Field")
       ax2.set_ylabel("Latitude")
       ax2.set_xlabel("Year")

    if showMagFields:
       #plot the time series for eachc latitude in a separate graph
       df.plot(x='Year',y=dCols, ylabel='microTesla',title="Solar Magnetic Field",figsize=(8, 6))
    plt.show()
   

