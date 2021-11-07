# -*- coding: utf-8 -*-
"""
Part of Semester Project at ICT4SM EPFL lab, autumn semester 2020
Maxime Gardoni
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import csvImporter
from bootstrap import BootstrapParent
sns.set_theme()
from computeMetrics import *
import pickle
from time import sleep

# the choosen NN architecture, here : 1 input, 20 hidden nodes, 1 output
architecture= [nn.Linear(1, 20), nn.LeakyReLU(), nn.Linear(20, 1)]

#retrain the net, if False, will use a previously stored one
RETRAIN = True

#Short computation of the performance over only 1 noise
SHORT_PERFORMANCE_TEST=True

for NoiseID in range(1,5): #for all the observation noise
    NoiseType= f"Noise_{NoiseID}"
    dataset=csvImporter.importDataset("../dataset")[NoiseType]

    if RETRAIN:
        parent= BootstrapParent(20, architecture)
        parent.trainParent(dataset, 0.3, 800)
        f = open(f"./bootstrap.pk", 'wb') #store the trained model for potential future usage
        pickle.dump(parent, f) #store the model itself
        f.close()
    else:
        f = open("./bootstrap.pk", 'rb')
        parent=pickle.load(f)
        f.close()

    #%% calculation on test Data
    if SHORT_PERFORMANCE_TEST:
        meanTrain, stdTrain = parent.predict(dataset["Train_dataset"]["x"])
        meanTest, stdTest = parent.predict(dataset[f"212{NoiseID}"]["x"])

        #analysis of the standard deviation over all children
        print(f"short performance summary for noise 212{NoiseID}")
        print(f"BEFORE drift: mean of std {stdTest[0:99].mean()} after drift : mean of std {stdTest[100:].mean()}")

        plt.plot(stdTest.detach().numpy(), label=f"{NoiseType}",alpha=0.6)
        plt.xlabel('sample number')
        plt.ylabel('Width of prediction')
        plt.title(f"Width of prediction for noise 212{NoiseID}")
        plt.legend()
        plt.show()

    #computation of all the drift metrics
    metricsDF = computeMetrics(dataset, parent, safety=1.2, smoothingWindow=5)

    #Saving the metrics to an excel sheet
    metricsDF.to_excel(f"../metrics_output/{NoiseType} Metrics.xlsx" )
    print(f"finished for {NoiseType} and saved to excel")
    sleep(1)#wait a bit so the user can read logs