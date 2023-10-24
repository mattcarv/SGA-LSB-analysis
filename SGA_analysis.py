import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.table import Table

file = Table.read('/home/mdocarm/Downloads/SGA-2020.fits')
df = file.to_pandas()


