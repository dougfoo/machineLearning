# loading important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
 
# Loading the data
DATA_DIR = '.'
FILE_NAME = 'diamonds.csv'
data_path = os.path.join(DATA_DIR, FILE_NAME)
diamonds = pd.read_csv(data_path)
print(diamonds.shape)


