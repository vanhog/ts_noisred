import re
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from numpy import datetime64
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from scipy import signal as sg
import geopandas as gpd

import sentinel1helper as sh
from sqlalchemy.engine import row
from _sqlite3 import Row


in_file = '/media/hog/fringe1/dev/data/testn.csv'
in_file = '/media/hog/fringe1/dev/data/tl5_l2b_044_02_0001-0200.csv'
#in_file = '/media/hog/fringe1/sc/MSCDATA/Roenne-Overview/aoi_msc_gpk/tl5_l2b_aoi_msc_gpkg.gpkg'
#in_layer = 'tl5_a_044_01_mscaoi'
#in_file = '/media/nas01/hog/sc/sc_data/BBD_TL5/schleswig-holstein/l2b_schleswig-holstein_clipped.gpkg'
#in_layer = 'ASCE_044_02'
#df = sh.read_geofile(in_file, layer=in_layer, engine='pyogrio')
df = pd.read_csv(in_file)


#gdf = gpd.read_file('/media/hog/fringe1/sc/lab/BBD/BBD TL4/BBDTL4-Deiche/Nordstrand/Charakteristische_Zeitreihen/BBDTL4-ASCE-Nordstrand-v2.3-015_01_plus_ts.gpkg')
print(len(df))
print(df.head())

dt_dats, dats, nodats = sh.get_numpy64_dates(df.columns)

# padding to 6-day-cycle
dt_dats_padded = []
old_date = dt_dats[0]
while old_date <= dt_dats[-1]:
    dt_dats_padded.append(old_date)
    old_date += 6
    
# get dates as number sequence for numerical reasons
dt_dats_asDays          = (dt_dats - dt_dats[0]).astype('float')
dt_dats_padded_asDays   = (dt_dats_padded - dt_dats_padded[0]).astype('float')


# filter data
query_list = [27534195, 27534443]

df_part = df[df['PS_ID'].isin(query_list)]

for idx, row in df_part.iterrows():
     this_ps = row 
     this_ts = this_ps[dats]
     this_mv = this_ps[nodats]