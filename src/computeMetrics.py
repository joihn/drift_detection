# -*- coding: utf-8 -*-
"""
Part of Semester Project, autumn semester 2020
Maxime Gardoni
"""


import numpy as np
import pandas as pd

def movingAverageFIR(a, n=5) :
    """
    smoothing average using only the a past value = finite impulse response
    :param a: original signal
    :param n: number of sample to take into account
    :return: smoothed signal
    """
    temp= np.convolve(a, np.ones(n)/n, mode="valid")
    return np.concatenate((np.ones(n-1)*temp[0], temp ))

def computeMetrics(dataset, parent, safety, smoothingWindow):
    """
    Will compute the drift metrics
    :param dataset: dataset dictionnary
    :param parent: parent object
    :param safety: multiplication float
    :param smoothingWindow: number of sample for the smoothing
    :return:
    """
    metrics={}
    meanTrain, stdTrain = parent.predict(dataset["Train_dataset"]["x"])
    threshold = movingAverageFIR(stdTrain.detach().numpy(), smoothingWindow ).max() * safety
    DRIFTSTART= 100

    for noiseVariant in dataset:
        if noiseVariant != "Train_dataset":
            #get the prediction
            meanTest, stdTest = parent.predict(dataset[noiseVariant]["x"])

            stdTestSmooth = movingAverageFIR(stdTest.detach().numpy())
            driftFlagArrray= stdTestSmooth>threshold
            fP= driftFlagArrray[0:DRIFTSTART].sum() #false positive
            fN=(driftFlagArrray[DRIFTSTART:] == 0).sum()

            # lag of detection
            if np.max(driftFlagArrray[DRIFTSTART:]) == 0:
                lagOfDetection= np.array([np.nan])
            else:
                lagOfDetection= np.argmax(driftFlagArrray[DRIFTSTART:])

            metrics[noiseVariant]={
                "fP":fP.item(),
                "lagOfDetection": lagOfDetection.item(),
                # "fN":fN.item()
            }
    #convert our nested dictionnary into a panda dataframe for easier analysis
    metricsDF=pd.DataFrame.from_dict({(i): metrics[i]
                                      for i in metrics.keys()},
                                     orient='index')
    metricsDF.index.names = ['noiseVariant']
    return metricsDF
