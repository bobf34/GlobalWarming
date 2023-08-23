# Global Warming
A simple, but surprisingly accurate model for predicting global temperatures.  Volcanic activity and climate oscillations will cause the actual temperature to fluctuate around the prediction.
<br><br>
The model is a hybrid model predicting global temperatures using using sunspots and CO2 concentrations.
![Plot](./images//TempPrediction.png)
![Plot](./images//with_El_Niño_Events.png)


### Model Description
[Brief Sunspot/CO2 Model Description](hybridmodel.md)

### To Run
Download the python program files.

Setup the environment:
<br>
pip install -r requirements.txt

Run __tempPredict.py__  for the sunspot/CO2 model

### Changing the Sunspot/CO2 model
There are several different preconfigured models in a comment block.  Copy the desired model *parms* dictionary and replace (or place below) the parms dictionary located just below the comment block.  You can also create your own model by adjusting the parameters of an existing model.
<br><br>
Image (png) files showing predictions for a few of the models have been uploaded into this codespace. Click on the filename to view.

### Selecting Sunspot/CO2 Plots
A variable called *showExtra* can be configured to show the model, or the prediction error.
<br>
Set variable *showSpectrums* to True for plots of the temperature and sunspot spectrums.

### Required Datasets
The first time you run the program it will automatically download the required datasets.

__WARNING:__ Your results may change, or may not match results shown here as the data sets are constantly being updated and revised.jjj

### Misc
When __getTempSunspotData.py__ is run as a stand-alone program it will plot temperature and sunspot data.
<br><br>
When __getSynopticData.py__ is run as a stand-alone program it will plot a synoptic chart, butterfly diagram, and magnetic field data.  This program is not used by the model.

### Data Credits
Sunspot Data: WDC-SILSO, [Royal Observatory of Belgium, Brussels](https://www.sidc.be/silso/datafiles)
<br> 
Global Temperature Anomaly Data HadCRUT5: [Met Office](https://www.metoffice.gov.uk/hadobs/hadcrut5)<br>
Global Temperature Anomaly Data NOAA: [NOAA]( https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies)
<br>
Wilcox Solar Observatory Synoptic Charts and Data [WSO](http://wso.stanford.edu/synopticl.html)
<br>
El Niño Southern Oscillation (ENSO) data: [NOAA](https://psl.noaa.gov/enso/)
