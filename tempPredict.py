import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import pdb
from scipy.fft import fft,ifft,fftshift
from getTempSunspotData import getTempSunspotData


def rms(x):
    return np.sqrt(np.mean(np.square(x)))

def getCO2model(scale=1):
    # model was fit to combined Maui and ice-core datasets. For years beteen 1880 and now, returns a value
    # between zero and scale.  The shape is based on the log of CO2 concentrations
    polyco2 = [ 3.73473114e-07, -2.12676942e-03,  4.03870559e+00, -2.55744899e+03]
    co2Model = np.poly1d([x*scale for x in polyco2])
    return co2Model

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

def lifeTheUniverseAndEverything(df_ss,x_ss,gain=0.8):
    # The model needs a partial null at the 42 year cycle.  Unfortunately the impulse response would make the model
    # filter very long. Too long to use, in fact. So, instead, a sinusoid is injected to cancel most of the signal 
    # before it gets to the model.  Not a great solution, but it works.
    f = 0.024169921875  #~ 1/42  frequency of the 42 year sunspot cycle
    [g,phi] = getToneMagPhase(df_ss,x_ss,f)
    x42 = g*np.cos(2*np.pi*f*df_ss.Year+ phi)
    x_ss -= gain*x42
    return x_ss

def getToneMagPhase(df_ss,x_ss,f):
    fftSize = 8192*2
    fbin = int(f*fftSize/12)
    padLen = fftSize-len(df_ss)
    x = np.pad(x_ss,(0,padLen))
    X = fft(x)
    g = 2*np.abs(X[fbin])/len(df_ss)  #fftSize/len(df_ss) * X[fbin]/(fftSize/2)
    phi = -np.angle(X[fbin])
    return [g,phi]
    


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
    return(model)

def computeTemps(df_temp,x_model,co2comp,advance,MA=1):
    # compensate the temperature data from the dataset
    co2Model = getCO2model(co2comp)
    co2 = co2Model(df_temp.Year)  #the model
    x_temp = df_temp.Temperature - co2

    #fit the predicted temps for scale and offset agains the co2 compensated actual temps
    ig = int((firstValidYear - df_temp.Year[0])*12)
    #ig = 20*12
    if advance>0:
               [scale,offset] =  np.polyfit(x_model[ig:-12*advance],x_temp[ig:],1)
    else:
               [scale,offset] =  np.polyfit(x_model[ig:],x_temp[ig:],1)
    x_model_comp =  scale*x_model + offset
    x_model_comp += co2Model(t_model)

    #rmserr = rms(df_temp.Temperature[ig:] - x_model[ig:len(df_temp)])
    err = df_temp.Temperature[ig:] - x_model_comp[ig:len(df_temp)]
    if MA>0:
      errrMA = np.convolve(err,np.ones(MA*12)/(MA*12),mode='valid')
      rmserr = rms(errrMA)
    else:
      rmserr = rms(err)
    return [x_temp,x_model_comp,co2Model,rmserr]



###################################################################################################

parms = {
        'modType':'RECT2',  # choices are RECT,RECT2, or NOTCH
        'rectW':99,    #width in years of the moving average nom:99
        'rectW2':11.1, #width of the short RECT, ignored for RECT and NOTCH model types
        'MA': 3,       #moving average in years for temperature plots and rms error computation
        'M42':True,    #compensate for the 42-year cycle
        'advance':15,  #amount in yearss to forward shift model output (i.e. years of prediction beyond sunspot data
        'co2comp':-1, #amount of co2 compensation degC  set negative for automatic selection
        }

'''
About the model types
                                                                                           _____
   RECT is a single boxcar shape which is the functional equivalent of a moving average  _|     |_
                                                                                            ____
   RECT2 is the convolution of two RECTs, a long and a short which is a RECT with ramps   _/    \_                                                              
   NOTCH is a RECT convolved with a bandstop filter centered on the 11-year sunspot cycle.  The notch suppresses
   energy at the frequency of the 11-year sunspot cycle, which is also the purpose of RECT2 when set to l1 years. The notch 
   just accomplishes this with a bit more finesse.

#BASIC MODEL 99-year moving average predicts 13 years into future
parms = {'modType':'RECT','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 'advance':13, 'co2comp':0}  

#Improved Model 1, adds 11 year moving average predicts 8 years into future
parms = {'modType':'RECT2','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 'advance':8, 'co2comp':0}  

#Improved Model 2, replaces 11 year moving average with notch filter
parms = {'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':False, 'advance':0, 'co2comp':0.0}  

#BEST Model adds compensation for 42 year sunspot cycle 
parms = {'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':0.0}  

#Best Model with fixed CO2 compensation
parms = {'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':0.3}  

#Best Model with search for best CO2 compensation
parms = {'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':-1}  
'''
#Best Model with fixed CO2 compensation
parms = {'modType':'NOTCH','rectW':99, 'rectW2':11.1, 'MA':3, 'M42':True, 'advance':0, 'co2comp':0.3}  

showModel=True
firstValidYear = 1900 #ignore the data before 1900 when fitting and computing error

# Get the temperature and sunspot datasets
[df_temp,df_ss] = getTempSunspotData(useLocal = True, plotData=False)

#create a 3yr moving average of the temperature data for plotting purposes
tempMA = np.convolve(df_temp.Temperature,np.ones(parms['MA']*12)/(parms['MA']*12),mode='valid')
t_tempMA = np.arange(len(tempMA))/12+df_temp.Year[0]+parms['MA']/2

# modify the amplitude and offset of the sunspot data to make it easier to work with
x_ss = df_ss.Sunspots.values
x_ss -= np.mean(x_ss)
if parms['M42']:
   x_ss = lifeTheUniverseAndEverything(df_ss,x_ss,0.8)  #attenuate the 42 year sunspot cycle

# get the sunspot-to-temperature model
model = getModel(parms['modType'],parms['rectW'], parms['rectW2'])

# Use the model to predict the temperature 
x_model = np.convolve(x_ss,model,mode='valid')

advance =  parms['advance']  #for convinience

#truncate the data prior to the first year in df_temp
x_model = x_model[-(len(df_temp)+12*advance):]
#create a time index
t_model = np.arange(len(x_model))/12+df_temp.Year[0]

# Search for the co2 compensation which produces the lowest RMS error
bestCO2comp=0
bestRMSerr = 1e6
comps = np.arange(40)/40
rmserrs = []
for co2Comp in comps:
    [x_temp,x_model_comp,co2Model,rmserr] = computeTemps(df_temp,x_model,co2Comp,advance,parms['MA'])
    rmserrs.append(rmserr)
    if rmserr < bestRMSerr:
        bestRMSerr = rmserr
        bestCO2comp = co2Comp

if parms['co2comp']>=0:  #use the co2comp param instead of best fit for final plots
    bestCO2comp= parms['co2comp']

# recompute the prediction using the best compensation
[x_temp,x_model_comp,co2Model,rmserr] = computeTemps(df_temp,x_model,bestCO2comp,advance,parms['MA'])

# Plot the results

fig = plt.figure(constrained_layout=True)
fig.suptitle(str(parms))
gs = fig.add_gridspec(3, 3)
if showModel:
   if parms['co2comp']<0:  #use the co2comp param instead of best fit for final plots
     ax_ss = fig.add_subplot(gs[0,0:2])
     ax_fit = fig.add_subplot(gs[0,2])
   else:
     ax_ss = fig.add_subplot(gs[0,0:])

   ax_temp = fig.add_subplot(gs[1:,0:])
else:
   if parms['co2comp']<0:  #use the co2comp param instead of best fit for final plots
       ax_fit = fig.add_subplot(gs[0,0:])
       ax_temp = fig.add_subplot(gs[1:,0:])
   else:
       ax_temp = fig.add_subplot(gs[0:,0:])

if showModel:
   ax_ss.set_title('Sunspots -  Source: WDC-SILSO, Royal Observatory of Belgium, Brussels')
   ax_ss.plot(df_ss.Year,df_ss.Sunspots,'.7',label='Sunspots')

   ax_model = ax_ss.twinx()
   plotmodel = np.pad(model,1)  #zero pad for plotting purposes
   ax_model.plot((np.arange(len(plotmodel))-len(plotmodel))/12+1880-parms['advance'],plotmodel,'b',label='Model')
   ax_model.legend()

if parms['co2comp']<0:  #use the co2comp param instead of best fit for final plots
   ax_fit.set_title('Model Error vs CO2 Compensation ('+str(parms['MA'])+'Yr MA)')
   ax_fit.set_ylabel('RMS error °C')
   ax_fit.set_xlabel('CO2 compensation °C')
   ax_fit.grid()
   ax_fit.plot(comps,rmserrs)

#tempMA = np.convolve(x_temp,np.ones(36)/36,mode='valid')
ax_temp.set_title('Temperature Anomalies ')
ax_temp.plot(df_temp.Year,df_temp.Temperature,'.8',label='NOAA Global Temp Anomaly')
ax_temp.plot(t_tempMA,tempMA,'r',label='Temp '+str(parms['MA'])+' Yr Moving Average')
ax_temp.plot(t_model,co2Model(t_model),'.1',dashes=[4,4],label='CO2 Compensation: '+str(bestCO2comp)+'°C')
#ax_temp.plot(t_model,x_model+co2Model(t_model),'b',label='Sunspot + CO2 Model')
ax_temp.set_ylabel('°C')
ax_temp.set_xlabel('Year')
ax_temp.plot(t_model,x_model_comp,'b',label='Sunspot + CO2 Model Prediction')
ax_temp.grid()
ax_temp.legend()

plt.show()
