# Global Warming
A simple, but surprisingly accurate model for predicting global temperatures using using sunspots and CO2 concentrations. Volcanic activity and climate oscillations will cause the actual temperature to fluctuate around the prediction.

![Plot](./TempPrediction.png)

### To Run
Download both py files
<br>
Run tempPredict.py

### Changing the model
There are several different preconfigured models in a comment block.  Copy the desired model *parms* dictionary and replace (or place below) the parms dictionary located just below the comment block.  You can also create your own model by adjusting the parameters of an existing model.
<br><br>
Image (png) files showing predictions for a few of the models have been uploaded into this codespace. Click on the filename to view.

### Selecting Plots
A variable called *showExtra* can be configured to show the model, or the prediction error.
<br>
Set variable *showSpectrums* to True for plots of the temperature and sunspot spectrums.

### Required Datasets
The first time you run the program it will automatically download the two required datasets.

### Brief Model Description
[Description of Model and Plots](plots.md)
### Detailed Model Description and Validation
[Model description and validation (pdf)](https://localartist.org/media/CutlerModelDescription.pdf)

### Data Credits
Sunspot Data: WDC-SILSO, [Royal Observatory of Belgium, Brussels](https://www.sidc.be/silso/datafiles)
<br>
Global Temperature Anomaly Data: [NOAA]( https://www.ncei.noaa.gov/access/monitoring/global-temperature-anomalies/anomalies)

