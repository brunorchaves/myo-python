Work in progress

## Myo Predictor

Using the api myo-python (add link later) to train a recurrent neural network to predict hand gestures
<p align="center"><img src="results/demo.gif"  width="250">

The code paramDataset.py at myo_predictor/gesture_predictor/ is used to capture signals from the Myo armband and generate .csv files to make datasets for further analysis.
The code trainGesture.py at myo_predictor/gesture_predictor/ is a neural network implemenation designed to be trained with the .csv files generated with paramDataset.py and to create a model.
The code runModel.py, as the name says, is used to run the classification model in real time with the armband.


How to merge .csv into only one (windows):
1. Browse to the folder with the CSV files.
2. Hold down Shift, then right-click the folder and choose Copy as path.
3. Open the Windows Command prompt.
4. Type cd, press Space, right-click and select Paste, then press Enter.
5. Type copy *.csv combined-csv-files.csv and Press Enter.
