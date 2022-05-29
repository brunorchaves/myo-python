import numpy as np
from series2gaf import GenerateGAF

# create a random sequence with 200 numbers
# all numbers are in the range of 50.0 to 150.0
random_series = np.random.uniform(low=50.0, high=150.0, size=(200,))

# set parameters
timeSeries = list(random_series)
windowSize = 50
rollingLength = 10
fileName = 'demo_%02d_%02d'%(windowSize, rollingLength)

# generate GAF pickle file (output by Numpy.dump)
GenerateGAF(all_ts = timeSeries,
            window_size = windowSize,
            rolling_length = rollingLength,
            fname = fileName)

from series2gaf import PlotHeatmap

gaf = np.load('demo_50_10_gaf.pkl', allow_pickle=True)
PlotHeatmap(gaf)