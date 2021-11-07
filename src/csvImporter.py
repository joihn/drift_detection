"""
Part of Semester Project at ICT4SM EPFL lab, autumn semester 2020
Maxime Gardoni
"""

import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import torch

def importDataset(pathToDataset):
    """
    Will read the .csv file, detrend them (cf report §3.3) and return a nested dictionary containing pandas dataframe
    """
    dataset={}

    for noiseType in listdir(pathToDataset):
        datasOneNoise={}
        for file in listdir(f"{pathToDataset}/{noiseType}"):
            df=pd.read_csv(f"{pathToDataset}/{noiseType}/{file}",  header=None)
            df.index=["x","y"]
            pdF=df.T
            datasOneNoise[file[0:-4]]= {
                "x":torch.tensor(pdF["x"].values).float(),
                "y": torch.tensor(pdF["y"].values).float()
            }
        dataset[noiseType]=datasOneNoise

    #detrending
    for noiseType in dataset:
        meanX= dataset[noiseType]["Train_dataset"]["x"].mean()
        meanY= dataset[noiseType]["Train_dataset"]["y"].mean()
        stdX= dataset[noiseType]["Train_dataset"]["x"].std()
        stdY= dataset[noiseType]["Train_dataset"]["y"].std()
        for noiseVariant in dataset[noiseType]:
            dataset[noiseType][noiseVariant]["x"].sub_(meanX).div_(stdX)
            dataset[noiseType][noiseVariant]["y"].sub_(meanY).div_(stdY)

    return dataset
#%%
if __name__ == "__main__":
    #small test for showing capabilities
    pathToDataset = "../dataset"
    dataset= importDataset(pathToDataset)

    #%%
    plt.plot(dataset["Noise_1"]["Train_dataset"]["x"], label="x")
    plt.plot(dataset["Noise_1"]["Train_dataset"]["y"], label= "y")
    plt.legend()
    plt.show()

