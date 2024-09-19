from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA

num_of_ts = 15
in_file = '/media/hog/fringe1/dev/data/testn.csv'
in_file = '/media/hog/fringe1/dev/data/tl5_l2b_044_02_0001-0200.csv'
df = pd.read_csv(in_file)

data = df.loc[:,'date_20150406':'date_20211230'].to_numpy()
print(data.shape)
print(df.head())

#plt.imshow(data, cmap="gray")

plt.plot(data[num_of_ts,:])


#scaler = StandardScaler()
x = data# = scaler.fit_transform(data) # normalizing the feature

scaler = StandardScaler()
x = scaler.fit_transform(x) # normalizing the features
    
pca = PCA(n_components=100)
pca_result= pca.fit_transform(x)
print(type(pca))
print(len(pca_result), type(pca_result), pca_result.shape)
PCA_reverse = pca.inverse_transform(pca_result)
unnormalized = scaler.inverse_transform(PCA_reverse) 
plt.plot(unnormalized[num_of_ts,:])

print(pca.explained_variance_ratio_.cumsum())



diff_sig = [i-j for i,j in zip(unnormalized[num_of_ts,:], data[num_of_ts,:])]
sig_diff = [i-j for i,j in zip(data[num_of_ts,0:-1], data[num_of_ts,1:])]

plt.figure()
plt.plot(diff_sig)

plt.figure()
plt.plot(sig_diff)
plt.show()