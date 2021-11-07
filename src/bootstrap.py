# -*- coding: utf-8 -*-
"""
Part of Semester Project, autumn semester 2020
Maxime Gardoni
"""

import pickle

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as Data

from copy import deepcopy

from datetime import datetime
import csvImporter


sns.set_theme()

class BootstrapParent():
    """
    The NN parent class, which will contains N children (N small neural nets as child)
    """
    def __init__(self, nChildren, layerList):
        """
        generate the children and put them in a list
        :param nChildren: the desired number of children
        :param layerList: the desired architecture as a list of sequential layer
        """
        self.children=[]
        #getting the number of input of the NN
        self.nInput=np.nan
        firstLay=True
        for layer in layerList:
            if hasattr(layer, "in_features"):
                if firstLay:
                    self.nInput=layer.in_features
                    firstLay=False

        for n in range(nChildren):
            #create the children
            layerList=deepcopy(layerList)#deep copy the model otherwise they are all linked to the same object
            for l in layerList:
                if hasattr(l, 'reset_parameters'):
                    l.reset_parameters() #each model has a random starting point, so they will converge to different minima
            self.children.append(self.BootstrapChild(layerList))

    def adaptDataX(self, x):
        """
        Transform the data dimension for it to be compatible with the NN.
        If the NN has multiple inputs, it will create a nSample x nWindow array using a moving window
        example: [1,2,3,4,5,6], prepared for 3 inputs, will become

        [
        1,2,3;
        2,3,4;
        3,4,5;
        4,5,6;
        ]

        :param x: input data as 1D pytorch array
        :return: transformed data, 2D pytorch array
        """

        if self.nInput==1:
            return x.view(-1,1)
        else:
            xExtended=torch.empty(x.shape[0]-self.nInput+1, self.nInput)

            for i in range(xExtended.shape[0]):
                xExtended[i]= x[i: i + self.nInput]
            return xExtended

    def adaptDataY(self, y):
        """
        Transform the data dimension for it to be compatible with the NN.
        If the NN has multiple inputs, it can't do predicitons for the first nInputs-1 sample, and those are therefore dropped

        :param y: pytorch array
        :return: pytorch 2d array
        """

        if self.nInput==1:
            return y.view(-1,1)
        else:
            return y[self.nInput-1:].view(-1,1)

    def assignTrainToChildren(self, dataset, subSamplePercentage):
        """
        For each child, get a random subSammple of the dataset and assign it for later training

        :param dataset: dictionnary: the dataset itself
        :param subSamplePercentage: float ∈ ]0,1[ , the normlised size of the subSample set

        """
        if subSamplePercentage<=0 or subSamplePercentage>=1:
            raise AssertionError

        nSelectedSample=round(len(dataset["Train_dataset"]["x"]) * subSamplePercentage)
        nIgnoredSample = len(dataset["Train_dataset"]["x"]) - nSelectedSample
        for c in self.children:
            sequenceStart= np.random.randint(0, nIgnoredSample)
            c.trainData={
                'x': self.adaptDataX(
                        dataset["Train_dataset"]["x"][0:nSelectedSample]
                    ),
                'y': self.adaptDataY(
                        dataset["Train_dataset"]["y"][0:nSelectedSample]
                    )}

            c.validData={
                'x': self.adaptDataX(
                    dataset[list(dataset.keys())[0]] # validatiation set is the first 100 sample of the first noise variant
                        ["x"][0:99]),
                'y': self.adaptDataY(
                    dataset[list(dataset.keys())[0]]
                        ["y"][0:99]
                    )}


    def trainParent(self, dataset, subSamplePercentage, epoch):
        """
        train the parent (will train the children one by one)
        :param trainData:
        :param subSamplePercentage:
        :return:
        """
        self.assignTrainToChildren(dataset, subSamplePercentage)
        for i,c in enumerate(self.children):
            print(f"training child number {i} ")
            finalLoss=c.trainChild(epoch, len(self.children)==1)



    def predict(self, x) -> (np.float, np.float):
        """
        Compute the NN prediction, that is, for each sample of X, predict a Y and it's standard deviation (incertitude measurement)
        :param x: 1D pytorch array
        :return: tuple:(prediction of Y, incertitude measurement)
        """
        xAdapted = self.adaptDataX(x)

        #store the prediction in a matrix
        predArray=torch.empty((len(self.children), xAdapted.shape[0]))

        for i, c in enumerate(self.children): #for each child
            predArray[i, :]= c(xAdapted).view(-1) # do a point prediction for each sample of X

        # To get the mean prediction, an average over all the children is done
        # To get incertitude, a standard deviation over all children is done
        return predArray.mean(dim=0), predArray.std(dim=0)

            
    class BootstrapChild(nn.Sequential):
        def __init__(self, layerList):
            """
            extent the sequential class to add some more method to it
            each child will have an object of this class

            :param layerList: the
            """
            super().__init__( *layerList)  #supercharging the nn.Sequential class,

            #give a name to the child for easy reference when grid search
            self.name=""
            firstLay=True
            for layer in layerList:
                if hasattr(layer, "in_features"):
                    if firstLay:
                        self.nInput=layer.in_features
                        firstLay=False
                    self.name=f"{self.name} {layer.in_features}"
            self.name=f"{self.name} 1"
    
        def trainChild(self,epoch, computeTest):
            """
            train an invidual child
            :param epoch: number of train iteration
            :param computeTest: boolean, if enabled we will compute the testLoss for convergence monitoring
            :return: the train loss
            """
            #with SGD optimizer and no minibatch
            # print(f"trainChild is training child {self.childName}")
            optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
            loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

            for e in range(epoch):
                prediction = self(self.trainData["x"])
                optimizer.zero_grad()   #
                loss = loss_func(prediction, self.trainData["y"])

                if computeTest:
                    with torch.no_grad():
                        prediction = self(self.validData["x"])
                        testLoss = loss_func(prediction, self.validData["y"])
                        # if e==epoch-1:
                        print(f"                                      epoch n {e}, name{self.name}, loss {round(loss.item(), 3)} testloss {round(testLoss.item(),3)}")
                else:
                    if e%100==0 or e==epoch-1:
                        print(f"epoch n {e}, loss {round(loss.item(), 3)}")
                loss.backward()         # backpropagation, compute gradients
                optimizer.step()        # apply gradients
            return loss

        