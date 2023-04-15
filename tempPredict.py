import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import pdb
from scipy.fft import fft,ifft,fftshift
from getTempSunspotData import getTempSunspotData


def rms(x):
    #returns root-mean-square scalar for vector (array) x
    return np.sqrt(np.mean(np.square(x)))

def getCO2model(scale=1):
    # model was fit to combined Maui and ice-core datasets. For years between 1880 and Nov 2020, returns a value
    # between zero and scale.  The shape is based on the log of CO2 concentrations
    # use of a model allows the co2 compensation to extend into the future beyond existing co2 measurements
    polyco2 = [ 3.73473114e-07, -2.12676942e-03,  4.03870559e+00, -2.55744899e+03]
    co2Model = np.poly1d([x*scale for x in polyco2])
    return co2Model

def co2Model(co2comp,year):
    # compute the co2 model over time-values in year
    return getCO2model(co2comp)(year)

def elevenYrNotch():
    # Improves accuracy over 11-year moving average
        fc = .092 
        bw = .06
        ln=27 #years
        #build the bandstop filter 
        y=signal.firwin(ln*12+1,[2*(fc-bw)/12,2*(fc+bw)/12])
        # Modify the filter to limit the amount of attenuation in the stopband
        g = 1.3
        y[int(len(y)/2)] *=g 
        y /=g
        return y

def lifeTheUniverseAndEverything(df_ss,x_ss,gain=0.8,f = 0.024169921875):
    # The model needs a partial null at the 42 year cycle.  Unfortunately, the impulse response would make the model
    # filter very long. Too long to use, in fact. So, instead, a sinusoid is injected to cancel most of the signal 
    # before it gets to the model.  Think of it active cancelling, like noise cancelling headphones
    #f = 0.024169921875  #~ 1/42  frequency of the 42 year sunspot cycle
    dt = 1/12  #data is sampled monthly, or 12 times/year
    results = dft(x_ss,dt,[f])
    x42 = idft(results,len(x_ss),1/12)
    x_ss -= gain*x42
    return x_ss

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


def getModel(modelType,rectW, rectW2):
    # Build the sunspot-to-temperature model
    if modelType == 'NOTCH':
        W2 = elevenYrNotch()
        model = np.convolve(np.ones(rectW*12),W2,mode='full')

    elif modelType == 'RECT2':
        W2 = np.ones(int(rectW2*12))
        model = np.convolve(np.ones(rectW*12),W2,mode='full')

    else: #RECT
        model = np.ones(rectW*12)

    model /= np.sum(model)    
    if modelType == 'NONE':
        model *= 0
    return(model)

def getDataInFitRegion(df_temp,df_tempMA,df_model,firstValidYear,co2comp=1):
    if firstValidYear > df_tempMA.Year[0]:
       fitStartYear = firstValidYear
    else:
       firstFitYear = df_tempMA.Year[0]
    lastFitYear = df_tempMA.Year.iloc[-1]

    idxMA =np.where((df_tempMA.Year>=fitStartYear)&(df_tempMA.Year <=lastFitYear))[0]
    idxModel =np.where((df_model.Year>=fitStartYear)&(df_model.Year <=lastFitYear))[0]

    tma = df_tempMA.Temperature[idxMA].values
    model = df_model.Temperature[idxModel].values
    year = df_tempMA.Year[idxMA]

    co2 = co2Model(co2comp,year)
    return [tma,model,co2,year]

def fitSunspotModelToTemp(df_temp,df_tempMA,df_model,co2comp,firstValidYear):
    '''
     fit gm and c to minimize the error in err = (tma-co2) - gm*model + c
     where: model is the sunspot model prediction
            co2 is the co2 model prediction scaled by co2comp
            tma is the Moving averaged Temperature
            gm is gain and c is a constant offset
      
    In matrix form the error = [tma-co2] - [gm c][model 1]
                             or TMA - AX where A= [gm c]
    
    The fit will be over the years of: greater of firstValid and the first year in the moving average
    through the last year in the moving average temperature.  
    '''

    [tma,model,co2,year] = getDataInFitRegion(df_temp,df_tempMA,df_model,firstValidYear,co2comp)
    tma_comp = tma-co2  #subtract the co2 model compensation from the temperature

    X =  np.vstack([model,np.ones(len(model))]).T
    [gm,c] = np.linalg.lstsq(X, tma_comp, rcond=None)[0]

    rmserr = rms(tma_comp - np.dot(X,[gm,c]))   #rms err is computed over the fit interval
    fitParms = {'gainSS':gm, 'gainCO2':co2comp,'offset':c,'rmserr':rmserr}
    #print('fitParms: '+str(fitParms))
    return fitParms 
    
def fitBothModelsToTemp(df_temp,df_tempMA,df_model,firstValidYear):
    '''
     fit gm,gc, and c to minimize the error in err = tma - gm*model + gc*co2 + c
     where: model is the sunspot model prediction
            co2 is the co2 model prediction
            tma is the Moving averaged Temperature
            gm and gc are gains and c is a constant offset
      
    In matrix form the error = [tma'] - [gm gc c][model' co2' 1]
                             or TMA - AX where A= [gm gc c]
    
    The fit will be over the years of: greater of firstValid and the first year in the moving average
    through the last year in the moving average temperature.  
    '''

    [tma,model,co2,year] = getDataInFitRegion(df_temp,df_tempMA,df_model,firstValidYear)
    X =  np.vstack([model,co2,np.ones(len(model))]).T
    [gm,gc,c] = np.linalg.lstsq(X, tma, rcond=None)[0]

    rmserr = rms(tma - np.dot(X,[gm,gc,c]))   #rms err is computed over the fit interval
    fitParms = {'gainSS':gm, 'gainCO2':gc,'offset':c,'rmserr':rmserr}
    #print('fitParms: '+str(fitParms))
    return fitParms 

def computeCombinedModelOutput(fitParms, dftemp,df_model):
    co2 = co2Model(fitParms['gainCO2'],df_model.Year)
    model = fitParms['gainSS']*df_model.Temperature+fitParms['offset']
    temp = model+co2
    df_model_comb = pd.DataFrame({'Year':df_model.Year,
                                'Temperature':temp,
                                'modelPredict':model,
                                'co2predict':co2})

    idx =np.where((df_model_comb.Year>=dftemp.Year[0])&(df_model_comb.Year <=dftemp.Year.iloc[-1]))[0]
    error = dftemp.Temperature.values - df_model_comb.Temperature[idx].values
    df_err = pd.DataFrame( {'Year': dftemp.Year, 'error': error })
    if False: #for debug use
        df_model_comb.plot(x='Year')
        df_err.plot(x='Year')
        plt.show()
    
    return [df_model_comb,df_err]


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


###################################################################################################

'''
parms = {
        'modelName':'MyModel', #Appears on plots
        'fname':'predictionResults.csv', #filename to save results, use empty string for no save
        'modType':'RECT2',  # choices are RECT,RECT2, NOTCH, or NONE for CO2 only
        'rectW':99,    #width in years of the moving average nom:99
        'rectW2':11.1, #width of the short RECT, ignored for RECT and NOTCH model types
        'MA': 3,       #moving average in years for temperature plots and rms error computation
        'M42':True,    #compensate for the 42-year cycle
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

#CO2 only model
parms = {'modelName':'CO2 Only Model','fname':'',
         'modType':'NONE','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':-1}  

#BASIC MODEL 99-year moving average predicts 13 years into future
parms = {'modelName':'Basic: 99', 'fname':'',
         'modType':'RECT','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 'advance':13, 'co2comp':0}  

#Improved Model 1, adds 11-year moving average predicts 8 years into future
parms = {'modelName':'Model 1: 99-11','fname':'',
         'modType':'RECT2','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 'advance':8, 'co2comp':0}  

#Improved Model 2:  Best for future prediction Adds compensation for 42
parms = {'modelName':'Model 2: 99-11-42 Best for future prediction','fname':'',
         'modType':'RECT2','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':8, 'co2comp':0}  

#Improved Model 3, replaces 11 year moving average with notch filter, no 42-year comp
parms = {'modelName':'Improved Model 3: 99-N','fname':'',
         'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 'advance':0, 'co2comp':0.0}  

#BEST Model adds Notch and compensation for 42 year sunspot cycle No CO2
parms = {'modelName':'Best Sunspot-Only Model: 99-N-42','fname':'',
         'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':0.0}  

#Best Model with Notch, 42-year and fixed CO2 compensation
parms = {'modelName':'Best Model: 99-N-42-fixed CO2','fname':'',
         'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':0.28}  

#Best Model with Notch, 42-year and search for best CO2 compensation
parms = {'modelName':'Best Model: 99-N-42 CO2 search','fname':'',
         'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':-1}  
'''
#Best Model with fixed CO2 compensation
parms = {'modelName':'Best Model: 99-N-42-fixed CO2','fname':'foo2.csv',
         'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':0.28}  

showExtra='error'  #'model' plots the model over the sunspot data used for the first prediction in 1880
                   #'error' plots the error over the prediction
                   # False  for no extra plot

showModelName = True  #displays the model name
showParms = False     #displays the parms variable 
optimalCO2 = True     #if co22comp is negative, simultaneously fit sunspot and co2 models, otherwise search and plot search outpu
                      #In otherwords, to see the shape of the fit, set this variable to False and co2comp to -1

firstValidYear = 1895 #ignore the data before 1895 when fitting and computing error, 
                      #the earliest global temperature and/or sunspot data may be not be that accurate.
splitYear=2000        # new satellites with better temperature sensors were launched around 2000 used for error plot only
                      # to show RMS error over two different time periods

# Get the temperature and sunspot datasets
[df_temp,df_ss] = getTempSunspotData(useLocal = True, plotData=False)

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

# modify the offset of the sunspot data to make it easier to work with
x_ss = df_ss.Sunspots.values
x_ss -= np.mean(x_ss)

if parms['M42']:
   x_ss = lifeTheUniverseAndEverything(df_ss,x_ss,0.9)  #attenuate the 42 year sunspot cycle

# get the sunspot-to-temperature model
model = getModel(parms['modType'],parms['rectW'], parms['rectW2'])

# Use the model to predict the temperature.  Convolve the model with the sunspot data
x_model = np.convolve(x_ss,model,mode='valid')

advance =  parms['advance']  #convenience variable

#truncate the data prior to the first year in df_temp
x_model = x_model[-(len(df_temp)+12*advance):]

#create a time index
t_model = np.arange(len(x_model))/12+df_temp.Year[0]

# combine the time index with the uncompensated prediction
df_model = pd.DataFrame( {'Year':t_model,'Temperature':x_model})


if parms['co2comp']>=0:
    fitParms = fitSunspotModelToTemp(df_temp,df_tempMA,df_model,parms['co2comp'],firstValidYear)
    bestCO2comp = fitParms['gainCO2']
elif optimalCO2: #simultaneously fit sunspot and co2 model predictions to temperature data
    fitParms = fitBothModelsToTemp(df_temp,df_tempMA,df_model,firstValidYear)
    bestCO2comp = fitParms['gainCO2']
else: # Search for the co2 compensation which produces the lowest RMS error
    searchResolution = 0.02  #degC
    numSteps = int(1/searchResolution)
    bestCO2comp=0
    bestRMSerr = 1e6
    comps = np.arange(numSteps)/numSteps
    rmserrs = []
    for idx,co2Comp in enumerate(comps):
        fitParms = fitSunspotModelToTemp(df_temp,df_tempMA,df_model,co2Comp,firstValidYear)
        rmserr = fitParms['rmserr']
        rmserrs.append(rmserr)
        if rmserr < bestRMSerr:
            bestRMSerr = rmserr
            bestCO2comp = co2Comp
    fitParms = fitSunspotModelToTemp(df_temp,df_tempMA,df_model,bestCO2comp,firstValidYear)

# scale the prediction and add co2 compensation to model
[df_model_comb,df_err]=computeCombinedModelOutput(fitParms,df_tempMA,df_model)

if parms['fname']:
    saveResultsCSV(df_temp,df_tempMA,df_model_comb,df_err,fname=parms['fname'])

###### PLOT THE RESULTs #####
fig = plt.figure(figsize=(10, 6), constrained_layout=True)
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
   ax_extra.plot(df_err.Year.values[idx1],rms1*np.ones(len(idx1[0])),'r',dashes=[1,1],label='RMS:'+'{:.3f}'.format(rms1))
   ax_extra.plot(df_err.Year.values[idx2],rms2*np.ones(len(idx2[0])),'r',dashes=[4,1],label='RMS:'+'{:.3f}'.format(rms2))
   ax_extra.legend()


if showCO2search:  #plot the co2 optimization
   ax_fit.set_title('Model Error vs CO2 Compensation ('+str(parms['MA'])+'Yr MA)')
   ax_fit.set_ylabel('RMS error °C')
   ax_fit.set_xlabel('CO2 compensation °C')
   ax_fit.grid()
   ax_fit.plot(comps,rmserrs)

#plot the temperatures
ax_temp.set_title('Temperature Anomalies ')
ax_temp.plot(df_temp.Year,df_temp.Temperature,'.8',label='NOAA Global Temp Anomaly')
ax_temp.plot(df_tempMA.Year,df_tempMA.Temperature,'r',label='Temp '+str(parms['MA'])+' Yr Moving Average')
#ax_temp.plot(t_model,co2Model(bestCO2comp,t_model),'.1',dashes=[4,4],label='CO2 Compensation: '+str(bestCO2comp)+'°C')
ax_temp.plot(df_model_comb.Year,df_model_comb.co2predict,'.1',dashes=[4,4],label='CO2 Compensation: '+'{:1.3f}'.format(bestCO2comp)+'°C')
ax_temp.set_ylabel('°C')
ax_temp.set_xlabel('Year')
errStr = ': RMS Error:'+'{:.3f}'.format(fitParms['rmserr'])+'°C'
if parms['modType'] == 'NONE':
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='CO2 Model Only Prediction'+errStr)
elif bestCO2comp == 0:
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='Sunspot Model Only Prediction'+errStr)
else:
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='Sunspot + CO2 Model Prediction'+errStr)
ax_temp.grid()
ax_temp.legend()
plt.show()
