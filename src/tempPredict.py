import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import pdb
from scipy.fft import fft,ifft,fftshift
from getTempSunspotData import getTempSunspotData
from getENSOdata import ENSO
#from sklearn.linear_model import LinearRegression  #imported later for weighted fitting.  Optional.

#parameters which control the design of the 11-year notch
#fc = center frequency, bw = bandwidth, g controls depth of notch. Attenuation decreases as g increases min =1
f11parms1 = {'fc':.0933,'bw':.061,'g':1.05}  #lowest RMS error and greatest suppression of the eleven year cycle
f11parms2 = {'fc':.0933,'bw':.061,'g':1.55}   #lowest RMS error for data beyond splitYear (more accurate satellite temps)
f11parms3 = {'fc':.0933,'bw':.061,'g':1.3}   #compromise 

#parameters affecting 42-year cycle suppression
#fc = center frequency, g controls amount of suppression. Suppression increases with G, max = 1
f42parms1 = {'fc':[0.024169921875],'g':[0.69]}

tempDataSource = 'HC5'

firstDispYear = 1880
lastDispYear = 2030
firstValidYear = 1900 # Ignore the data before 1895 when fitting and computing error, 
                      # The earliest global temperature and/or sunspot data may be not be that accurate.
splitYear = 1997      # New satellites with better temperature sensors started launchng in 1998 (NOAA-15) which might
                      # explain the sudden change in the variance of the error.  On the error plot, the RMS error
                      # is computed both before and after splitYear.  When a weighted LMS fit is used, the statistics
                      # of the error before and after this date are used to weight the fit.
'''
parms = {
        'modelName':'MyModel', #Appears on plots
        'fname':'predictionResults.csv', #filename to save results, use empty string for no save
        'modType':'RECT2',  # choices are RECT,RECT2, NOTCH, or NONE for CO2 only
        'rectW':99,    #width in years of the moving average nom:99
        'rectW2':11.1, #width of the short RECT, ignored for RECT and NOTCH model types
        'MA': 3,       #moving average in years for temperature plots and rms error computation
        'M42':True,    #compensate for the 42-year cycle
        'optimalCO2': True  #Option override of global variable.  See variable for description
        'f42parms': f42parms1  #parameters which control the 42-year cycle suppression
        'f11parms': f11parms1, #parameters which control the design of the 11-year notch filter
        'weightedFit':False  #Optional override of global variable. See variable for description
        'extraWeight':1  #Optional raises weight by power of extraWeight
        'advance':15,  #amount in yearss to forward shift model output (i.e. years of prediction beyond sunspot data
        'co2comp':-1,  #amount of co2 compensation degC  set negative for automatic selection
        }

About the model types
                                                                                           _____
   RECT is a single boxcar shape which is the functional equivalent of a moving average  _|     |_
                                                                                            ____
   RECT2 is the convolution of two RECTs, a long and a short which is a RECT with ramps   _/    \_                                                              
   NOTCH is a RECT convolved with a bandstop filter centered on the 11-year sunspot cycle.  The notch suppresses
   energy at the frequency of the 11-year sunspot cycle, which is also the purpose of RECT2 when set to l1 years. The notch 
   just accomplishes this with a bit more finesse.

   NOTE:  The RMS errors shown below may not match your results as the sunspot and global temperature datasets are modified and updated

#>>>>> CO2 only model. 
parms = {'modelName':'CO2 Only Model','fname':'CO2_only.csv', 'modType':'NONE',
         'rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':0, 'co2comp':-1}  

#>>>>> Core 99-Year Moving Average Model
parms = {'modelName':'Core 99-Year Moving Average Model', 'fname':'Core_99.csv',
         'modType':'RECT','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':13.5, 'co2comp':0}  

#>>>>> Core 99-year plus 11-year Moving averages
parms = {'modelName':'Model 99-11','fname':'99_11.csv',
         'modType':'RECT2','rectW':99, 'rectW2':11, 'MA':3, 'M42':False, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':9.0, 'co2comp':0}  


#>>>>> Core 99-year plus 11-year Moving averages plus 42-yeer compensation
parms = {'modelName':'99-11-42','fname':'99_11_42.csv',
         'modType':'RECT2','rectW':98, 'rectW2':10, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':9.9, 'co2comp':0}  

#>>>>> 99-11-42 with CO2:  <<<<<<<<<<<<<. 
parms = {'modelName':'99-11-42 CO2','fname':'99_11_42_C.csv',
         'modType':'RECT2','rectW':98, 'rectW2':10, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':10.0, 'co2comp':-1}  

#>>>>> 98-Notch
parms = {'modelName':'98-Notch','fname':'98_N.csv',
         'modType':'NOTCH','rectW':98, 'rectW2':11, 'MA':3, 'M42':False, 
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':1, 'co2comp':0.0}  

#>>>>> 98-Notch-42 
parms = {'modelName':'98-Notch-42','fname':'98_N_42.csv',
        'modType':'NOTCH','rectW':98, 'rectW2':11, 'MA':3, 'M42':True,'tempDataSource':'HC5',
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0.0, 'weightedFit':False, 'extraWeight':4, 'co2comp':0}  

#>>>>> 98-Notch-42 Search for CO2
parms = {'modelName':'98-Notch-42 search CO2','fname':'98_N_42_CS.csv','tempDataSource':'HC5',
         'modType':'NOTCH','rectW':98, 'rectW2':11, 'MA':3, 'M42':True, 'optimalCO2': False,
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0, 'co2comp':-1}  

#>>>>> 98-Notch-42 CO2 using NOAA temperature data
parms = {'modelName':'98-Notch-42 NOAA','fname':'98_N_42_C_NOAA.csv',
         'modType':'NOTCH','rectW':98, 'rectW2':11, 'MA':3, 'M42':True,'tempDataSource':'NOAA',
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':1.4, 'weightedFit':True, 'extraWeight':4, 'co2comp':-1}  

#>>>>> 98-Notch-42 CO2 using HadCRUT5 temperature data
parms = {'modelName':'98-Notch-42 CO2 HadCRUT5','fname':'98_N_42_C_HC5.csv',
        'modType':'NOTCH','rectW':98, 'rectW2':11, 'MA':3, 'M42':True,'tempDataSource':'HC5',
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0.4, 'weightedFit':False, 'extraWeight':4, 'co2comp':-1}  
'''
#Active Model Copy from comment lock above and place below
parms = {'modelName':'98-Notch-42 CO2 HadCRUT5','fname':'98_N_42_C_HC5.csv',
        'modType':'NOTCH','rectW':98, 'rectW2':11, 'MA':3, 'M42':True,'tempDataSource':'HC5',
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0.4, 'weightedFit':False, 'extraWeight':4, 'co2comp':-1}  


# NOTE:  firstValidYear and splitYear are defined before the models
saveResults = False  #If true and fname is defined in parms, the output results are saved into a CSV file
showSpectrums = False  #Create a separate plot window showing temperature and sunspot spectrums. Default False

showExtra='error'  # 'model' plots the model over the sunspot data used for the first prediction in 1880
                   # 'error' plots the error over the prediction
                   #  Use empty string '' for no extra plot

showVolcanos = False
volcanos = [1963+2/12, 1982+4/12, 1991+6/12]
volcanoTxt = 'Agung, El Chichón, and Pinatubo volcanic eruptions'

showEnsoEvents = False
ensoThresh = 2.0

showModelName = True  # Displays the model name in the plots. Default True
showParms = False     # Ddisplays the parms variable. Default False
optimalCO2 = True     # If True and co2comp is negative, simultaneously fit sunspot and co2 models, otherwise sweep 
                      # the CO2 level, fit the sunspot prediction, identify the CO2 level producing the minimum error and plot.
                      # Note: The CO2 and RMS results may vary slightly with this setting.

weightedFit = False   # Perform a weighted LMS fit (only if sklearn is installed)  Default: False
extraWeight = 1       # 0.5 for weighting of 1/stdDev, 1.0 for weighting of 1/variance  2 for pow(1/var,2), etc
                      # The larger the number, the more heavily weighted the fit is post splitYear

#Allow these globals to be overridden in parms dictionary
if 'optimalCO2' in parms:
   optimalCO2 = parms['optimalCO2']
if 'weightedFit' in parms:
   weightedFit = parms['weightedFit']
if 'extraWeight' in parms:
   extraWeight = parms['extraWeight']
if 'tempDataSource' in parms:
    tempDataSource = parms['tempDataSource']
if 'firstValidYear' in parms:
    firstValidYear = parms['firstValidYear']


###################################################################################################

def rms(x):
    #returns root-mean-square scalar for vector (array) 
    # so many choices....
    #  return np.sqrt(np.mean(np.square(x)))
    #  return np.sqrt(np.var(x))
    return np.std(x)

def getCO2model(scale=1):
    # For years between 1880 and Nov 2020, returns a value  between 0.09*scale and scale.  
    # The polynomial is a fit to the temperature data, but the model returns predictions
    # very close to the log of co2 contributions.
    # This model allows the co2 compensation prediction to extend into the future
    #polyco2 = [ 3.03495678e-07, -1.71295381e-03,  3.22327145e+00, -2.02202272e+03] # .09 - 1 degC from 1880- Nov 2020
    polyco2 = [ 3.51809168e-07, -1.98563900e-03,  3.73638417e+00, -2.34401630e+03]  # 0 - 1.1 degC from 1880-Feb 2023
    co2Model = np.poly1d([x*scale for x in polyco2])
    return co2Model

def co2Model(co2comp,year):
    # compute the co2 model over time-values in year
    return getCO2model(co2comp)(year)

def elevenYrNotch(f11parms = {'fc':.0933,'bw':.061,'g':1.3}):
    # Improves accuracy over 11-year moving average
    fc = f11parms['fc']
    bw = f11parms['bw']
    g =  f11parms['g']  #useful range 1.05 (less detail, lower RMS error) to ~1.55 (better match beyond year splitYear)
    ln=27 #27 years is the longest possible filter without loosing the ability to predict to the end of the sunspot data


    #build the bandstop filter 
    y=signal.firwin(ln*12+1,[2*(fc-bw)/12,2*(fc+bw)/12])

    # Modify the filter to limit the amount of attenuation in the stopband
    y[int(len(y)/2)] *=g  #scale the center bin impulse (increase) 
    y /=g                 #scale the entire filter (decrease)
    return y

def lifeTheUniverseAndEverything(df_ss,x_ss,f42parms={'fc':[0.024169921875],'g':[0.69]}):
    # The model needs a partial null at the 42 year cycle.  Unfortunately, the impulse response would make the model
    # filter very long. Too long to use, in fact. So, instead, a sinusoid is injected to cancel most of the signal 
    # before it gets to the model.  Think of it as active cancelling, like noise cancelling headphones
    #f = 0.024169921875  #~ 1/42  frequency of the 42 year sunspot cycle
    dt = 1/12  #data is sampled monthly, or 12 times/year
    fc = f42parms['fc'] # list of frequencies
    g = f42parms['g']   # list of gains for each frequency
    results = dft(x_ss,dt,fc)
    for i,a in enumerate(g):
        results[i][1] *= a
    x42 = idft(results,len(x_ss),1/12)
    return (x_ss - x42)

def dft(x,dt,freqList,t0=0):
    #compute the Fourier coef's at a list of frequencies
    t = np.arange(len(x))*dt + t0
    result = []
    for f in freqList:
       ex = np.exp(-1j*2*np.pi*f*t)
       coef = np.dot(x, ex)
       result.append([f,coef])
    return result    

def idft(fcoefs,pts,dt,t0=0):
    # synthesize a waveform from a list of frequencies and Fourier coef's
    x = np.zeros(pts)
    t = np.arange(pts)*dt + t0
    for [f,coef] in fcoefs:
         ex = np.exp(1j*2*np.pi*f*t)
         x += 2*np.real(coef*ex)/pts
    return x


def getModel(parms):
    # Build the sunspot-to-temperature model
    modelType = parms['modType']
    rectW = parms['rectW']
    rectW2 = parms['rectW2']

    W1 = np.ones(int(rectW*12))
    if modelType == 'NOTCH':  #Nomial: 11-year notch convoled with 99-year rect
        W2 = elevenYrNotch(parms['f11parms'])
        model = np.convolve(W1,W2,mode='full')

    elif modelType == 'RECT2': #Nominal 11-year rect convolved with 99-year rect
        W2 = np.ones(int(rectW2*12))
        model = np.convolve(W1,W2,mode='full')

    else: #RECT   #Nominal 99-year rect only.
        model = W1

    model /= np.sum(model)    
    if modelType == 'NONE':  #For CO2 only simpulations, set model to all zeros.
        model *= 0
    return(model)

def getDataInFitRegion(df_temp,df_tempMA,df_model,firstValidYear,co2comp=1):
    # returns the temperature and co2 data in the range from firstValidYear to last year with data.
    # The moving average ends early, so it determines the last year
    if firstValidYear > df_tempMA.Year[0]:
       fitStartYear = firstValidYear
    else:
       firstValidYear = df_tempMA.Year[0]
    lastFitYear = df_tempMA.Year.iloc[-1]

    idxMA =np.where((df_tempMA.Year>=fitStartYear)&(df_tempMA.Year <=lastFitYear))[0]
    dft = df_tempMA.iloc[idxMA]
    idxModel =np.where((df_model.Year>=fitStartYear)&(df_model.Year <=lastFitYear))[0]
    dfm = df_model.iloc[idxModel]
    idx = np.where(np.abs(np.subtract.outer(dft.Year.values,dfm.Year.values))<=1/12)


    #tma = df_tempMA.Temperature[idxMA].values
    tma = df_tempMA.iloc[idxMA].Temperature.values
    tma = dft.iloc[idx[0]].Temperature.values
    model = df_model.iloc[idxModel].Temperature.values
    model = dfm.iloc[idx[1]].Temperature.values
    year = df_tempMA.Year[idxMA]
    year = dft.iloc[idx[0]].Year

    co2 = co2Model(co2comp,year)
    return [tma,model,co2,year]

def getWeightedFit(err,year,splitYear,X,y):
    # Performes a weighted fit of the Model(s) to the temperature data and computes the error
    # returns the model and the error
    from sklearn.linear_model import LinearRegression
    #compute index variables based on splitYear
    region1 = np.where(year<splitYear)
    region2 = np.where(year>=splitYear)

    #construct a weighting vector
    w1 = 1/np.var(err[np.where(year<splitYear)]) #use of variance overfits
    w2 = 1/np.var(err[np.where(year>=splitYear)])
    w1 =pow(w1,extraWeight)
    w2 =pow(w2,extraWeight)
    wghts = np.concatenate((w1* np.ones(len(region1[0])),w2* np.ones(len(region2[0]))))
   
    #perform the fit and compute the error
    reg = LinearRegression().fit(X, y,wghts)
    err = (y - reg.predict(X)) #err is computed over the fit interval
    return [reg,err]

def fitSunspotModelToTemp(df_temp,df_tempMA,df_model,co2comp,firstValidYear,splitYear = 0):
    '''
     fit gm and c to minimize the error in err = (tma-co2) - gm*model + c
     where: model is the sunspot model prediction
            co2 is the co2 model prediction scaled by co2comp
            tma is the Moving averaged Temperature
            gm is gain and c is a constant offset
      
    In matrix form the error = [tma'] - [model' 1][gm c]'
                             or TMA - AX where X= [gm c]'
    
    The fit will be over the years of: greater of firstValid and the first year in the moving average
    through the last year in the moving average temperature.  
    '''

    [tma,model,co2,year] = getDataInFitRegion(df_temp,df_tempMA,df_model,firstValidYear,co2comp)
    tma_comp = tma-co2  #subtract the co2 model compensation from the temperature

    X =  np.vstack([model,np.ones(len(model))]).T
    # perform LMS fit
    [gm,c] = np.linalg.lstsq(X, tma_comp, rcond=None)[0]
    err = (tma_comp - np.dot(X,[gm,c])) #err is computed over the fit interval

    if weightedFit and splitYear > year.iloc[0]: #perform a weighted fit based on statistics of err
        try:
           #use error variance in first fit to perform a weighted fit
           [reg,err] = getWeightedFit(err,year,splitYear,X,tma_comp)
           gm = reg.coef_[0]
           c = reg.intercept_
           #err = (tma_comp - reg.predict(X)) #err is computed over the fit interval
        except Exception as e: 
            print(e)
            print("Warning: sklearn not installed")

    rmserr = rms(err)   #rms err is computed over the fit interval
    fitParms = {'gainSS':gm, 'gainCO2':co2comp,'offset':c,'rmserr':rmserr}
    return fitParms 
    
def fitBothModelsToTemp(df_temp,df_tempMA,df_model,firstValidYear,splitYear):
    '''
     fit gm,gc, and c to minimize the error in err = tma - gm*model + gc*co2 + c
     where: model is the sunspot model prediction
            co2 is the co2 model prediction
            tma is the Moving averaged Temperature
            gm and gc are gains and c is a constant offset
      
    In matrix form the error = [tma'] - [model' co2' 1][gm gc c]'
                             or TMA - AX where X= [gm gc c]'
    
    The fit will be over the years of: greater of firstValid and the first year in the moving average
    through the last year in the moving average temperature.  
    '''

    [tma,model,co2,year] = getDataInFitRegion(df_temp,df_tempMA,df_model,firstValidYear)
    # perform LMS fit
    X =  np.vstack([model,co2,np.ones(len(model))]).T
    [gm,gc,c] = np.linalg.lstsq(X, tma, rcond=None)[0]
    err = tma - np.dot(X,[gm,gc,c])
    
    if weightedFit and splitYear > year.iloc[0]: #perform a weighted fit based on statistics of err
        try:
           #use error variance in first fit to perform a weighted fit
           [reg,err] = getWeightedFit(err,year,splitYear,X,tma)
           gm = reg.coef_[0]
           gc = reg.coef_[1]
           c = reg.intercept_
        except Exception as e: 
            print(e)
            print("Warning: sklearn not installed")

    rmserr = rms(err)
    fitParms = {'gainSS':gm, 'gainCO2':gc,'offset':c,'rmserr':rmserr}
    return fitParms 

def computeCombinedModelOutput(fitParms, dftemp,df_model):
    # Combine the co2 and sunspot model predictions, compute the error and return in dataframes
    co2 = co2Model(fitParms['gainCO2'],df_model.Year)
    model = fitParms['gainSS']*df_model.Temperature+fitParms['offset']
    temp = model+co2
    df_model_comb = pd.DataFrame({'Year':df_model.Year,
                                   'Temperature':temp,
                                   'modelPredict':model,
                                   'co2predict':co2}).reset_index(drop=True)

    #startidx =np.where(df_model_comb.Year>=dftemp.Year[0])[0][0]
    #error = dftemp.Temperature.values - df_model_comb.iloc[startidx:startidx+len(dftemp)].Temperature.values
    #idx =np.where((df_model_comb.Year>=dftemp.Year[0])&(df_model_comb.Year <=dftemp.Year.iloc[-1]))[0]
    idx = np.where(np.abs(np.subtract.outer(dftemp.Year.values,df_model_comb.Year.values))<=1/24)
    error = dftemp.Temperature.loc[idx[0]].values - df_model_comb.Temperature.loc[idx[1]].values

    df_err = pd.DataFrame( {'Year': dftemp.Year.loc[idx[0]], 'error': error }).reset_index(drop=True)
    if False: #for debug use
        df_model_comb.plot(x='Year')
        df_err.plot(x='Year')
        plt.show()

    return [df_model_comb,df_err]

def getClosestIdx(years1,years2,dx=1/12):
     idx = np.where(np.abs(np.subtract.outer(years1,years2))<=dx)[0]
     locs = idx[np.where(np.diff(np.append([-10000],idx))>1)[0]]
     return locs

def  getMarkers (df,column,years):
     yearIdx = getClosestIdx(df.Year.values,years)
     return pd.DataFrame({'Year':years, 'Values': df[column].loc[yearIdx]})


def decYearToYrMo(decYr):
    #convert date of the form 1880.041667 to 188001  
    x = np.modf(decYr)
    return (x[1]*100 + (x[0]-1/24)*12+1).astype(int)

def saveResultsCSV(df_temp,df_tempMA,df_model_comb,df_err,fname):
    dft = df_temp.copy()
    dft['YrMo']=decYearToYrMo(dft.Year)
    dft.drop(['Year'], axis=1,inplace=True)

    dftMA = df_tempMA.rename(columns={"Temperature": "TempMovAvg"},inplace=False)
    dftMA['YrMo']=decYearToYrMo(dftMA.Year)
    dftMA.drop(['Year'], axis=1,inplace=True)

    dferr = df_err.copy()
    dferr['YrMo']=decYearToYrMo(dferr.Year)
    dferr.drop(['Year'], axis=1,inplace=True)

    df_output = df_model_comb.rename(columns={"Temperature": "PredictedTemp"},inplace=False)
    df_output['YrMo']=decYearToYrMo(df_output.Year)

    #merge on YrMo 
    df_output = pd.merge(df_output,dft,how='outer',on='YrMo')
    df_output = pd.merge(df_output,dftMA,how='outer',on='YrMo')
    df_output = pd.merge(df_output,dferr,how='outer',on='YrMo')

    #re-order the columns
    df_output = df_output[['YrMo','Year','Temperature','TempMovAvg','PredictedTemp','modelPredict','co2predict','error']]
    df_output.to_csv(fname,index=False)

def plotSpectrums(x_ss, df_temp,df_model_comb,bw=0.15,window='boxcar'):
    fftSize = 8192*4;  #make fft longer than data to allow better frequency bin resolution (doesn't improve RBW)
    f = np.arange(fftSize)*12/fftSize

    #compute window function
    #Note:  The boxcar, or uniform window provides the best frequency resolution and the worst amplitude accuracy
    if not window in ['boxcar','hann','hamming','cosine','flattop']:
        window='boxcar'
    w_ss = signal.windows.get_window(window,len(x_ss))
    w_temp = signal.windows.get_window(window,len(df_temp))
    w_model = signal.windows.get_window(window,len(df_model_comb))
    #scale the window for window correction factor, and for the upcoming FFT
    w_ss *= len(w_ss)/np.sum(w_ss) * 2/fftSize
    w_temp *= len(w_temp)/np.sum(w_temp) * 2/fftSize
    w_model *= len(w_model)/np.sum(w_model) * 2/fftSize

    #apply the window and zero pad out to the FFT size
    ss = np.pad(w_ss*x_ss,(0,fftSize-len(x_ss)))
    temp = np.pad(w_temp*df_temp.Temperature.values,(0,fftSize-len(df_temp.Temperature.values)))
    model = np.pad(w_model*df_model_comb.Temperature.values,(0,fftSize-len(df_model_comb.Temperature.values)))

    #compute the spectrums
    Xfss = fft(ss)
    Xftemp = fft(temp)
    Xftemp_m = fft(model)

    #plot the temperature spectrums
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(top=0.95,right=.9,left=.09,wspace=.5,hspace = .5)
    gs = fig.add_gridspec(2, 1)
    ax_spec = fig.add_subplot(gs[0, 0])
    ax_spec.set_title('Temperature Spectrums')
    ax_spec.plot(f,20*np.log10(np.abs(Xftemp_m)),label='Model Temp',color='b',dashes=(4,1)) #spectrum
    ax_spec.plot(f,20*np.log10(np.abs(Xftemp)),label='Temp',color='r') #spectrum
    ax_spec.grid()
    ax_spec.legend()
    ax_spec.set_ylabel('dB')
    ax_spec.set_xlabel('Frequency (1/Year)')
    ax_spec.set_xlim(0,bw)
    ax_spec.set_ylim(-75,-25)

    #plot the sunspot spectrum
    ax_spec1 = fig.add_subplot(gs[1, 0])
    ax_spec1.set_title('Sunspot Spectrum')
    ax_spec1.plot(f,20*np.log10(np.abs(Xfss)),label='Sunspot',color='m') #spectrum
    ax_spec1.grid()
    ax_spec1.set_ylabel('dB')
    ax_spec1.set_xlabel('Frequency (1/Year)')
    ax_spec1.set_xlim(0,bw)
    ax_spec1.set_ylim(-30,20)


############  BEGIN MAIN PROGRAM ####################

# Get the temperature and sunspot datasets

[df_temp,df_ss] = getTempSunspotData(useLocal = True, tempSrc = tempDataSource, plotData=False)

#safety check
if firstValidYear < df_temp.Year[0]:
   firstValidYear = df_temp.Year[0]

#create a 3yr moving average of the temperature data for plotting and prediction error purposes
if parms['MA']>0:
    tempMA = np.convolve(df_temp.Temperature,np.ones(parms['MA']*12)/(parms['MA']*12),mode='valid')
    t_tempMA = np.arange(len(tempMA))/12+df_temp.Year[0]+parms['MA']/2
    df_tempMA = pd.DataFrame( {'Year':t_tempMA,'Temperature':tempMA})
else:  #use unaveraged global temperature 
    df_tempMA = df_temp

# Get very strong ENSO (El Niño) events to add to plot
enso = ENSO(yearOffset=1/24)
df_enso_events = enso.getEvents(ensoThresh)
df_enso_events = df_tempMA.merge(df_enso_events,on='Year')

# modify the offset of the sunspot data to make it easier to work with
x_ss = df_ss.Sunspots.values
x_ss -= np.mean(x_ss)

if parms['M42']:
   x_ss = lifeTheUniverseAndEverything(df_ss,x_ss,parms['f42parms'])  #attenuate the 42 year sunspot cycle

# get the sunspot-to-temperature model
model = getModel(parms)

# Use the model to predict the temperature.  Convolve the model with the sunspot data
model_predict = np.convolve(x_ss,model,mode='valid')
t_model = df_ss.Year.iloc[-len(model_predict):]+parms['advance']

# combine the time index with the uncompensated prediction
df_model = pd.DataFrame( {'Year':t_model,'Temperature':model_predict})


if parms['co2comp']>=0:
    fitParms = fitSunspotModelToTemp(df_temp,df_tempMA,df_model,parms['co2comp'],firstValidYear,splitYear)
    bestCO2comp = fitParms['gainCO2']
elif optimalCO2: #simultaneously fit sunspot and co2 model predictions to temperature data
    fitParms = fitBothModelsToTemp(df_temp,df_tempMA,df_model,firstValidYear,splitYear)
    bestCO2comp = fitParms['gainCO2']
else: # Search for the co2 compensation which produces the lowest RMS error
    searchResolution = 0.02  #degC
    numSteps = int(1/searchResolution)
    bestCO2comp=0
    bestRMSerr = 1e6
    comps = np.arange(numSteps)/numSteps
    rmserrs = []
    for idx,co2Comp in enumerate(comps):
        fitParms = fitSunspotModelToTemp(df_temp,df_tempMA,df_model,co2Comp,firstValidYear,splitYear)
        rmserr = fitParms['rmserr']
        rmserrs.append(rmserr)
        if rmserr < bestRMSerr:
            bestRMSerr = rmserr
            bestCO2comp = co2Comp
    fitParms = fitSunspotModelToTemp(df_temp,df_tempMA,df_model,bestCO2comp,firstValidYear,splitYear)

# scale the prediction and add co2 compensation to model
[df_model_comb,df_err]=computeCombinedModelOutput(fitParms,df_tempMA,df_model)

if saveResults and parms['fname']:
    saveResultsCSV(df_temp,df_tempMA,df_model_comb,df_err,fname=parms['fname'])

if showSpectrums:
    plotSpectrums(x_ss, df_temp,df_model_comb)

###### PLOT THE RESULTs #####
#fig = plt.figure(figsize=(10, 6), constrained_layout=True)
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.88,right=.9,left=.09,wspace=.5,hspace = .5)
if showParms and showModelName and parms['modelName']:
    fig.suptitle(parms['modelName']+': '+str(parms))
elif showParms and parms['modelName']:
    fig.suptitle(str(parms))
elif showModelName:
    fig.suptitle(parms['modelName'])
gs = fig.add_gridspec(3, 3)

showCO2search = parms['co2comp']<0 and not optimalCO2

if showExtra in ['model','error'] : #show the model above the temps
   if showCO2search:  #show the co2 optimization next to the model or error plot
     ax_extra = fig.add_subplot(gs[0,0:2])
     ax_fit = fig.add_subplot(gs[0,2])
   else:
     ax_extra = fig.add_subplot(gs[0,0:])

   ax_temp = fig.add_subplot(gs[1:,0:])
else:
   if showCO2search:  #show the co2 optimization search above the temps
       ax_fit = fig.add_subplot(gs[0,0:])
       ax_temp = fig.add_subplot(gs[1:,0:])
   else: #only show the temperatures
       ax_temp = fig.add_subplot(gs[0:,0:])  

if showExtra == 'model': #plot the model positioned over the sunspot data used to produce the first 1880 prediction
   ax_extra.set_title('Sunspots -  Source: WDC-SILSO, Royal Observatory of Belgium, Brussels')
   ax_extra.plot(df_ss.Year,df_ss.Sunspots,'.7',label='Sunspots')

   ax_model = ax_extra.twinx()
   plotmodel = np.pad(model,1)  #zero pad for plotting purposes
   ax_model.plot((np.arange(len(plotmodel))-len(plotmodel))/12+1880-parms['advance'],plotmodel,'b',label='Model')
   ax_model.legend()

elif showExtra == 'error': # plot prediction error
   idx1 =np.where((df_err.Year>firstValidYear) & (df_err.Year<splitYear))
   rms1= rms(df_err.error.values[idx1])
   idx2 = np.where((df_err.Year>splitYear))
   rms2= rms(df_err.error.values[idx2])
   ax_extra.grid()
   ax_extra.set_ylabel('Error Magnitude °C')
   ax_extra.set_title('Temperature Prediction Error')
   ax_extra.plot(df_err.Year,np.abs(df_err.error),label = '|Error|')
   ax_extra.plot(df_err.Year.values[idx1],rms1*np.ones(len(idx1[0])),'r',dashes=[1,1],label='RMS:'+'{:.5f}'.format(rms1))
   ax_extra.plot(df_err.Year.values[idx2],rms2*np.ones(len(idx2[0])),'r',dashes=[4,1],label='RMS:'+'{:.5f}'.format(rms2))
   ax_extra.legend()
   ax_extra.set_xlim(firstDispYear, lastDispYear)


if showCO2search:  #plot the co2 optimization
   ax_fit.set_title('Model Error vs CO2 Comp')
   ax_fit.set_ylabel('RMS error °C')
   ax_fit.set_xlabel('CO2 compensation °C')
   ax_fit.grid()
   ax_fit.plot(comps,rmserrs)

#plot the temperatures
ax_temp.set_title('Temperature Anomalies ')
if tempDataSource == 'NOAA':
    ax_temp.plot(df_temp.Year,df_temp.Temperature,'.8',label='NOAA Global Temp Anomaly')
elif tempDataSource == 'HC5':
    ax_temp.plot(df_temp.Year,df_temp.Temperature,'.8',label='HadCRUT5 Global Temp Anomaly')
else:
    ax_temp.plot(df_temp.Year,df_temp.Temperature,'.8',label='??? Global Temp Anomaly')

ax_temp.plot(df_tempMA.Year,df_tempMA.Temperature,'r',label='Temp '+str(parms['MA'])+' Yr Moving Average')
#ax_temp.plot(t_model,co2Model(bestCO2comp,t_model),'.1',dashes=[4,4],label='CO2 Compensation: '+str(bestCO2comp)+'°C')
ax_temp.plot(df_model_comb.Year,df_model_comb.co2predict,'.1',dashes=[4,4],label='CO2 Compensation: '+'{:1.3f}'.format(bestCO2comp)+'°C')
ax_temp.set_ylabel('°C')
ax_temp.set_xlabel('Year')
ax_temp.set_xlim(firstDispYear, lastDispYear)
errStr = ': RMS Error:'+'{:.4f}'.format(fitParms['rmserr'])+'°C'
if parms['modType'] == 'NONE':
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='CO2 Model Only Prediction'+errStr)
elif bestCO2comp == 0:
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='Sunspot Model Only Prediction'+errStr)
else:
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='Sunspot + CO2 Model Prediction'+errStr)

if showVolcanos:
    df_volcanos = getMarkers(df_tempMA,'Temperature',volcanos)
    ax_temp.plot(df_volcanos.Year,df_volcanos.Values,'kD',label = volcanoTxt)

if showEnsoEvents:
    if ensoThresh < 0:
        lbltxt = 'ENSO Index<{:1.1f}'.format(ensoThresh)
    else:
        lbltxt = 'ENSO Index>{:1.1f}'.format(ensoThresh)

    ax_temp.plot(df_enso_events.Year,df_enso_events.Temperature,'ko',markersize=7,label=lbltxt)

ax_temp.grid()
ax_temp.legend()
plt.show()
