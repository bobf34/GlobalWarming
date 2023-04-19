import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import pdb
from scipy.fft import fft,ifft,fftshift
from getTempSunspotData import getTempSunspotData
#from sklearn.linear_model import LinearRegression  #imported later for weighted fitting.  Optional.

#parameters which control the design of the 11-year notch
#fc = center frequency, bw = bandwidth, g controls depth of notch. Attenuation decreases as g increases min =1
f11parms1 = {'fc':.0933,'bw':.061,'g':1.05}  #lowest RMS error and greatest suppression of the eleven year cycle
f11parms2 = {'fc':.0933,'bw':.061,'g':1.55}   #lowest RMS error for data beyond 2000 (more accurate satellite temps)
f11parms3 = {'fc':.0933,'bw':.061,'g':1.3}   #compromise 

#parameters affecting 42-year cycle suppression
#fc = center frequency, g controls amount of suppression. Suppression increases with G, max = 1
f42parms1 = {'fc':[0.024169921875],'g':[0.69]}

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
        'useRMSweight':True  #Optional override of global variable. See variable for description
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
parms = {'modelName':'1: CO2 Only Model','fname':'', 'modType':'NONE',
         'rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':0, 'co2comp':-1}  

#BASIC MODEL 99-year moving average predicts 13 years into future
parms = {'modelName':'2: Basic: 99-Year Moving Average Model', 'fname':'',
         'modType':'RECT','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':13, 'co2comp':0}  

#Improved Model 2, adds 11-year moving average predicts 8 years into future
parms = {'modelName':'3a: Model 99-11','fname':'',
         'modType':'RECT2','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':8, 'co2comp':0}  

#Same as 3a except slightly shorter rectW2, predicts 10 year.  Allows more 11-year energy in prediction
parms = {'modelName':'3b Model 99-9','fname':'',
         'modType':'RECT2','rectW':99, 'rectW2':9, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':10, 'co2comp':0}  


# Improved Model 3a:  Best for future prediction Adds compensation for 42
parms = {'modelName':'4a: : Best Model for Future Prediction without CO2','fname':'bestFuturePredict.csv',
         'modType':'RECT2','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':8, 'co2comp':0}  

#>>>>>>>>>  Model 4 with CO2:  <<<<<<<<<<<<<
parms = {'modelName':'4b: Winner: Best Model for Future Prediction with CO2','fname':'bestFuturePredict.csv',
         'modType':'RECT2','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms3, 'advance':8, 'co2comp':-1}  

#Improved Model 3, replaces 11 year moving average with notch filter, no 42-year comp
parms = {'modelName':'5: Improved Model 3: 99-N','fname':'',
         'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0, 'co2comp':0.0}  

#>>>>>>>>>>>  Model with best post balanced 2000 RMS error No CO2 compensation <<<<<<<<<<<<<<<<<
parms = {'modelName':'6: Best Sunspot-Only Lowest error after 2000','fname':'bestSunspotOnly.csv',
         'modType':'NOTCH','rectW':98.8, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0, 'weightedFit':True, 'useRMSweight':True, 'co2comp':0.0}  

# Interesting Overfit of post 2000 data.  Wrongly follows 1991 Pinatubo erruption recovery
parms = {'modelName':'7: Sunspot-Only Lowest error after 2000 (Overfit) ','fname':'bestSunspotOnly.csv',
         'modType':'NOTCH','rectW':98.8, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0, 'weightedFit':True, 'useRMSweight':False, 'co2comp':0.0}  

#>>>>>>>>>>>>  Model with lowest overall RMS error, less accruate post 2000  <<<<<<<<<<<<<<<<<<<
parms = {'modelName':'8: Winner -- Model with Lowest RMS error','fname':'LowestRmsModel.csv',
         'modType':'NOTCH','rectW':98.8, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms1, 'weightedFit':False, 'advance':0, 'co2comp':-1}  

#  Search demo uses f11parms3 compromise notch filter settings
parms = {'modelName':'9: Search for CO2','fname':'LowestRmsModel.csv',
         'modType':'NOTCH','rectW':98.8, 'rectW2':11.1, 'MA':3, 'M42':True, 'optimalCO2': False,
         'f42parms':f42parms1, 'f11parms':f11parms3, 'weightedFit':True, 'advance':0, 'co2comp':-1}  

#>>>>>>>>>>> Best Overall Model. Balances overall RMS error with post 2000 error <<<<<<<<<<<<<<<<<
parms = {'modelName':'10: Winner -- Overall Best Model','fname':'bestOverallModel.csv',
         'modType':'NOTCH','rectW':98.8, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0, 'weightedFit':True, 'useRMSweight':True, 'co2comp':0.28}  
'''
#Active Model Copy from comment lock above and place below

#>>>>>>>>>>> Best Overall Model. Balances overall RMS error with post 2000 error <<<<<<<<<<<<<<<<<
parms = {'modelName':'10: Winner -- Overall Best Model','fname':'bestOverallModel.csv',
         'modType':'NOTCH','rectW':98.8, 'rectW2':11.1, 'MA':3, 'M42':True, 
         'f42parms':f42parms1, 'f11parms':f11parms2, 'advance':0, 'weightedFit':True, 'useRMSweight':True, 'co2comp':0.26}  

saveResults = False  #If true and fname is defined in parms, the output results are saved into a CSV file

showExtra='error'  # 'model' plots the model over the sunspot data used for the first prediction in 1880
                   # 'error' plots the error over the prediction
                   #  Use empty string '' for no extra plot

showModelName = True  # Displays the model name in the plots. Default True
showParms = False     # Ddisplays the parms variable. Default False
optimalCO2 = True     # If True and co2comp is negative, simultaneously fit sunspot and co2 models, otherwise sweep 
                      # the CO2 level, fit the sunspot prediction, identify the CO2 level producing the minimum error and plot.
                      # Note: The CO2 and RMS results may vary slightly with this setting.

firstValidYear = 1895 # Ignore the data before 1895 when fitting and computing error, 
                      # The earliest global temperature and/or sunspot data may be not be that accurate.
splitYear = 2000      # New satellites with better temperature sensors were launched around 2000 which might
                      # explain the sudden change in the variance of the error.  On the error plot, the RMS error
                      # is computed both before and after splitYear.  When a weighted LMS fit is used, the statistics
                      # of the error before and after this date are used to weight the fit.
weightedFit = False   # Perform a weighted LMS fit (only if sklearn is installed)  Default: False
useRMSweight = True   # If False, use variance to weight the fit. Default True  Warning: Variance tends to overfit 

#Allow these globals to be overridden in parms dictionary
if 'optimalCO2' in parms:
   optimalCO2 = parms['optimalCO2']
if 'weightedFit' in parms:
   weightedFit = parms['weightedFit']
if 'useRMSweight' in parms:
   useRMSweight = parms['useRMSweight']


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
    polyco2 = [ 3.03495678e-07, -1.71295381e-03,  3.22327145e+00, -2.02202272e+03]
    co2Model = np.poly1d([x*scale for x in polyco2])
    return co2Model

def co2Model(co2comp,year):
    # compute the co2 model over time-values in year
    return getCO2model(co2comp)(year)

def elevenYrNotch(f11parms = {'fc':.0933,'bw':.061,'g':1.3}):
    # Improves accuracy over 11-year moving average
    fc = f11parms['fc']
    bw = f11parms['bw']
    g =  f11parms['g']  #useful range 1.05 (less detail, lower RMS error) to ~1.55 (better match beyond year 2000)
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
       firstFitYear = df_tempMA.Year[0]
    lastFitYear = df_tempMA.Year.iloc[-1]

    idxMA =np.where((df_tempMA.Year>=fitStartYear)&(df_tempMA.Year <=lastFitYear))[0]
    idxModel =np.where((df_model.Year>=fitStartYear)&(df_model.Year <=lastFitYear))[0]

    tma = df_tempMA.Temperature[idxMA].values
    model = df_model.Temperature[idxModel].values
    year = df_tempMA.Year[idxMA]

    co2 = co2Model(co2comp,year)
    return [tma,model,co2,year]

def getWeightedFit(err,year,splitYear,X,y):
    # Performes a weighted fit of the Model(s) to the temperature data and computes the error
    # returns the model and the error
    from sklearn.linear_model import LinearRegression
    #compute index variables based on splitYear
    region1 = np.where(year<splitYear)
    region2 = np.where(year>=splitYear)
   
    #select between weighting based on the RMS, or the variance
    fcn = rms  # np.var overfits
    if not useRMSweight:  #warning overfits
       fcn = np.var
      
    #construct a weighting vector
    w1 = fcn(err[np.where(year<splitYear)]) #use of variance overfits
    w2 = fcn(err[np.where(year>=splitYear)])
    wghts = np.concatenate((w1* np.ones(len(region1[0])),2* np.ones(len(region2[0]))))
   
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
      
    In matrix form the error = [tma-co2] - [gm c][model 1]
                             or TMA - AX where A= [gm c]
    
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
      
    In matrix form the error = [tma'] - [gm gc c][model' co2' 1]
                             or TMA - AX where A= [gm gc c]
    
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

############  BEGIN MAIN PROGRAM ####################

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
   x_ss = lifeTheUniverseAndEverything(df_ss,x_ss,parms['f42parms'])  #attenuate the 42 year sunspot cycle

# get the sunspot-to-temperature model
model = getModel(parms)

# Use the model to predict the temperature.  Convolve the model with the sunspot data
model_predict = np.convolve(x_ss,model,mode='valid')

advance =  parms['advance']  #convenience variable

#truncate the data prior to the first year in df_temp
model_predict = model_predict[-(len(df_temp)+12*advance):]

#create a time index
t_model = np.arange(len(model_predict))/12+df_temp.Year[0]

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

###### PLOT THE RESULTs #####
#fig = plt.figure(figsize=(10, 6), constrained_layout=True)
fig = plt.figure(figsize=(10, 6))
fig.subplots_adjust(top=0.88,right=.95,left=.09,wspace=.5,hspace = .5)
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


if showCO2search:  #plot the co2 optimization
   ax_fit.set_title('Model Error vs CO2 Comp')
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
errStr = ': RMS Error:'+'{:.4f}'.format(fitParms['rmserr'])+'°C'
if parms['modType'] == 'NONE':
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='CO2 Model Only Prediction'+errStr)
elif bestCO2comp == 0:
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='Sunspot Model Only Prediction'+errStr)
else:
    ax_temp.plot(df_model_comb.Year,df_model_comb.Temperature,'b',label='Sunspot + CO2 Model Prediction'+errStr)
ax_temp.grid()
ax_temp.legend()
plt.show()
