import pandas as pd
from hyperopt import tpe, hp, fmin
import numpy as np
import keras
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, concatenate
from random import shuffle
from keras import backend as K
import tensorflow as tf
from keras.models import Model
import gc
from sklearn.metrics import r2_score, mean_squared_error
import random
global singleFeatures
from scipy.cluster.hierarchy import linkage,cut_tree


# We need some global variables. Because couldn't pass them to Hyperopt.

global maxtRNALen
global tRNA
global ValSpecies
global allSpecies
global YTarget
global trainSpecies
global epochs
global sampleIm
global path1
global path2

# Tensorflow Configirations
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)


def elimData(dframe):
    trgt = dframe.values[:, 2]


    cnt1 = (np.abs(trgt - 28.5) > 0.5)
    cnt2 = (np.abs(trgt - 30.5) > 0.5)
    cnt3 = (np.abs(trgt - 37.5) > 0.5)

    cnt = np.logical_and(cnt1, cnt2)
    cnt = np.logical_and(cnt, cnt3)

    res = dframe[cnt]
    return pd.DataFrame(res.values)
def prepareData(isBacteriaOn, isArchaeaOn):
    # Takes two variables that indicates if we use archaea or bacteria
    # Returns read data as pandas data frame

    if isArchaeaOn and isBacteriaOn:
        dfA = readData(path1)
        dfA =elimData(dfA)
        dfB = readData(path2)
        dfB = elimData(dfB)
        d = concatDf([dfA, dfB])
        asize=dfA.shape[0]
        bsize=dfB.shape[0]
    elif isArchaeaOn:
        d = readData(path1)
        d = elimData(d)
        asize=d.shape[0]
        bsize=0
    elif isBacteriaOn:
        d = readData(path2)
        d = elimData(d)
        asize=0
        bsize=d.shape[0]

    return d, asize, bsize

def getTensor(Data, whichColumn):
    # Takes data as pandas data frame
    # Returns selected column as tensor. Selected column should have RNA sequences. Also returns max lenght of the RNA
    SEQs = getColumn(Data, Col=whichColumn)
    mxlen = len(max(SEQs.values[:,0],key=len))
    # Convert SEQs data frame to CNN compatible one-hot encoded.
    SEQTensor = getOneHotEncodedSet(SEQs, mxlen=mxlen)

    return SEQTensor, mxlen


def createModel(tRNALen, args):
    # takes 23S rRNA Lenght, 16S rRNA Lenght, tRNA Lenght and args dictionary
    # args dictionary contains CNN parameters. 1s are related to 23S, 2s are related to 16S, 3s are related to tRNA, 4s are related to tail of the model


    hidden_layer3 = args['hidden_layer4']
    drop_out3 = args['drop_out4']
    filter_size3 = args['filter_size3']
    kernel_size3 = args['kernel_size3']
    pool_size3 = args['pool_size3']
    strides3 = args['strides3']
    # tRNA model part input and output
    modelIntrna, modelOuttrna = buildModel(tRNALen, width=4, filterSize=filter_size3, kernelSize=kernel_size3,
                                         poolSize=pool_size3, denseSize=hidden_layer3, dropoutRate=drop_out3,
                                         strides=strides3)
    # read tail parameters
    hidden_layer4 = args['hidden_layer4']
    drop_out4 = args['drop_out4']


    generalOut = buildCombine(modelOuttrna, denseS=hidden_layer4, dropoutR=drop_out4)
    finalModel = Model(inputs=[modelIntrna], outputs=generalOut)

    return finalModel
def buildCombine(out1, out2=-1, out3=-1, denseS= 0, dropoutR=0):
    # Creates tail part of the model. If out2 and out3 are not passed model have a branch. If only out3 is not passed' model has 2 branches. Otherwise, iy has 3 branches
    # If one wants to build 2 branched model, should pass out1 and out2. out1 and out3 or out2 and out3 are not an option.
    if out2 == -1 and out3 ==-1: #one branch mode
        mymodel = out1
    elif out3 == -1: # 2 branch mode.
        mymodel= concatenate([out1, out2])
    else: #3 branch mode
        mymodel = concatenate([out1, out2, out3])

    mymodel = Dense(denseS, activation='relu')(mymodel)
    mymodel = Dropout(dropoutR)(mymodel)
    mymodel = Dense(denseS, activation='relu')(mymodel)
    mymodel = Dropout(dropoutR)(mymodel)
    mymodel = Dense(denseS, activation='relu')(mymodel)
    mymodel = Dropout(dropoutR)(mymodel)
    mOutput = Dense(1, activation='linear')(mymodel)
    return mOutput

    return mymodelOut
def readData(path, deli="\t", head=0):
    # Takes path and returns read data as pandas data frame
    # head indicates number of rows should be skipped for reading. 0 skips first row. -1 does not skip any row. Optional
    # deli shows delimmiter. Optional.

    return pd.read_csv(path, header=head,delimiter = deli)
def concatDf(dfList):
    # Concats list of data frames into one data frame
    df = dfList[0]
    for l in range(1, len(dfList)):
        df = pd.concat([df, dfList[l]], axis=0)
    return pd.DataFrame(df.values)

def filterDistMatrix(myUniqueSpecies, DM):
    #May be deleted! Not in use yet.
    specListinDistM= DM.columns.values
    res=np.zeros(DM.shape[0])
    myUniqueSpecies = ["\"{}\"".format(item) for item in myUniqueSpecies]
    myUniqueSpecies = ["'" + item.strip('"') + "'" for item in myUniqueSpecies]
    for s in myUniqueSpecies:
        if s in specListinDistM:
            res[np.where(s==specListinDistM)]=1
    res=np.array(res, dtype=bool)
    dumDM = DM.values[:,res]
    dumDM2 = dumDM[res,:]
    return specListinDistM[res], dumDM2

def getColumn(df, Col):
    # Takes a pandas data frame and returns a column
    res = df[df.columns[Col]]
    return pd.DataFrame(res.values)
def  getTarget(df,targetColumn):
    # Takes data frame and target column number
    # Returns target as array
    ogt = getColumn(df, Col=targetColumn).values
    ogt=np.array(ogt)
    OGT=[]
    # Not optimal. A better way will be added.
    for a in ogt:
        OGT.append(a[0])
    ogt= np.array(OGT)
    return ogt
def  measurePerformance(PR, real):
    # takes prediction and real arrays
    # Returns mean absolute error, pearson r, r squared and root mean squared error
    mae = np.mean(np.abs(np.array(PR) - np.array(real)))
    prsn = np.corrcoef(np.array(real), np.array(PR))[0,1]
    r2 = r2_score(np.array(real), np.array(PR))
    rmse = np.sqrt(mean_squared_error(np.array(real), np.array(PR)))
    return np.round(mae, 3), np.round(prsn, 3), np.round(r2, 3), np.round(rmse, 3)
def getOneHotEncodedSet(SEQs, mxlen, depth=4):
    # Takes sqquence list, and max lenght of sequences
    # Returns one hot encoded sequence list. Shorter sequences are 0 padded.
    one_hot_seqs = []
    for seq in SEQs.values[:, 0]:
        one_hot_seqs.append(one_hot_encoding(seq, mxlen))

    one_hot_seqs = np.array(one_hot_seqs)
    return  one_hot_seqs.reshape(one_hot_seqs.shape[0], depth, mxlen, 1)
def one_hot_encoding(seq, mx):
    # Takes a sequence and max lenght
    # Returns one hot encoded 2-dimensional array.
    dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    one_hot_encoded = np.zeros(shape=(4, mx))

    for i, nt in enumerate(seq):

        if nt.upper() == "A" or nt.upper() == "C" or nt.upper() == "G" or nt.upper() == "T" :
            one_hot_encoded[:,i] = dict[nt.upper()]
        else:
            continue
    return one_hot_encoded
def buildModel(window_size, width=4, filterSize=0, kernelSize=0, poolSize=0,denseSize=0, dropoutRate=0, strides=0):
    # Builds a convolutional branch and returns input and output node
    #output node is not sized 1! It's an array.
    mInput = Input(shape=(width, window_size, 1))

    #Convolution
    model = Conv2D(filterSize, kernel_size=(1, kernelSize), padding='valid', activation='relu',kernel_initializer='normal')(mInput)
    model = Conv2D(filterSize, kernel_size=(1, kernelSize), padding='same', activation='relu',kernel_initializer='normal')(model)
    model = MaxPooling2D(pool_size=(1, poolSize),strides=(1, strides), padding='same')(model)

    model = Conv2D(filterSize, kernel_size=(1, kernelSize), padding='same', activation='relu',kernel_initializer='normal')(model)
    model = Conv2D(filterSize, kernel_size=(1, kernelSize), padding='same', activation='relu',kernel_initializer='normal')(model)
    model = MaxPooling2D(pool_size=(1, poolSize), strides=(1, strides),padding='same')(model)

    model = Conv2D(filterSize, kernel_size=(1, kernelSize), padding='same', activation='relu',kernel_initializer='normal')(model)
    model = Conv2D(filterSize, kernel_size=(1, kernelSize), padding='same', activation='relu',kernel_initializer='normal')(model)
    model = MaxPooling2D(pool_size=(1, poolSize), strides=(1, strides),padding='same')(model)

    # Dense layers
    model = Flatten()(model)
    model = Dense(denseSize, activation='relu')(model)
    model = Dropout(dropoutRate)(model)
    mOutput = Dense(denseSize, activation='relu')(model)

    return mInput, mOutput

def performCNN( maxtRNALen, best_parameters, trainSpecies,testSpecies ):
    # This function is a copy of objective function. It can take all parameters so we do not need global variables
    # The other difference is this function returns a prediction and real values not a loss.
    global epochs
    global sampleIm
    # create model
    mymodel = createModel( maxtRNALen, best_parameters)

    # set some parameters
    batch_size = best_parameters['batch_size']
    beta1 = best_parameters['beta1']
    beta2 = best_parameters['beta2']
    learning_rate = best_parameters['learning_rate']

    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2)
    mymodel.compile(loss="mse", optimizer=adam, metrics=['mean_absolute_error'])

    # True-False array that shows training set
    TrainCont = (allSpecies == trainSpecies[0])
    for unqs in trainSpecies:
        TrainCont = np.logical_or(TrainCont, (allSpecies == unqs))



    PR = []
    REAL = []


    mymodel.fit(tRNA[TrainCont], YTarget[TrainCont].ravel(), batch_size=batch_size, epochs=epochs, verbose=1)

    for v in testSpecies:  # Get predictions on validation set.
        specContTest = (allSpecies == v)  # True-False array that shows testing set

        tey = YTarget[specContTest]
        tey = tey.ravel()
        PR.append(np.median(mymodel.predict(tRNA[specContTest]))) # Species OGT prediction is obtained as median of RNA related predictions
        REAL.append(tey[0])
    PR = np.array(PR)
    REAL = np.array(REAL)

    del mymodel
    gc.collect()
    K.clear_session()

    return PR, REAL

def writefile(P, R, S, pth):
    # P is predictions, R is real values, S is species array. Pth is path.
    for i in range(len(R)):
        with open(pth , "a") as myfile:
            mystr = S[i] +","+ str(R[i]) + "," + str(P[i]) + "\n"
            myfile.write(mystr)
    return

####################################   START     ##############################################################
####################################   INPUTS    ###############################################################
isBacteriaOn=1
isArchaeaOn=1

#predictor options

valRate = 0.05  # 0 to 1. Percentage of validation set
numberofChunks = 5 # To divide data set into groups after validation set is excluded.
n_clusters=10
#(Then numberofChunks -1 groups are used for training, 1 group is used for  testing. This process is repeated numberofChunks times.)

epochs=30 # CNN epochs
max_eval=3 # number of trials in hyper parameter optimization

    # Initializations

trnaSeqColumn=10 # in data file
targetColumn = 2 # in data file
speciesColumn = 1 # in data file

#Input files
path1 = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Archaea_RNAs_wOGT/Archaea_RNAs_wOGT.txt"
path2 = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Bacteria_RNAs_wOGT/Bacteria_RNAs_wOGT.txt"
pathDistance = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/allSpecies_taxid_phyloTree_dm.txt" # distance matrix
# provide path for output
path ="D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/regression_model-Output/PhylogeneticSplit/Final-Distance-"+str(isBacteriaOn)+"-archaea-"+str(isArchaeaOn)+"-isRandomSplit-0-"+"-chunks-"+str(numberofChunks)+"-ValRate-"+str(valRate)+".txt"

#Change parameters if there is a new optimized parameters

if isArchaeaOn==1 and isBacteriaOn==0: # only Archaea
    optimizedParams={'strides3': 8, 'pool_size3': 32, 'learning_rate': 0.001, 'kernel_size3': 32,
                               'hidden_layer4': 8, 'filter_size3': 32, 'drop_out4': 0, 'beta2': 0.94599515944805,
                               'beta1': 0.9644550506354505, 'batch_size': 64}
elif isArchaeaOn==0 and isBacteriaOn==1: #only Bacteria
    optimizedParams = {'strides3': 8, 'pool_size3': 3, 'learning_rate': 0.001, 'kernel_size3': 16, 'hidden_layer4': 128,
                       'filter_size3': 64, 'drop_out4': 0, 'beta2': 0.9856022493409639, 'beta1': 0.9663892064448255,
                       'batch_size': 16}
else: # Archaea and Bacteria together
    optimizedParams = {'strides3': 5, 'pool_size3': 32, 'learning_rate': 0.001, 'kernel_size3': 5, 'hidden_layer4': 32,
                       'filter_size3': 128, 'drop_out4': 0, 'beta2': 0.9468395195719862, 'beta1': 0.9310654774073976,
                       'batch_size': 32}

####################################end of input parameters#############################################################
##########################################################################################################################
##########################################################################################################################
numberofClustersinEachChunk=int(n_clusters/numberofChunks)

DM= readData(pathDistance)
# Read data from files
Data,ArcSize,BacSize = prepareData(isBacteriaOn, isArchaeaOn)

mySpecies = Data[Data.columns[speciesColumn]]
myUniqueSpecies = mySpecies.unique()
DMspecies, fDM = filterDistMatrix(myUniqueSpecies, DM)
linkage_matrix = linkage(fDM, "ward")
clusters =cut_tree(linkage_matrix, n_clusters=n_clusters).ravel()



tRNA, maxtRNALen = getTensor(Data, trnaSeqColumn)


# Get target OGTs
YTarget = getTarget(Data, targetColumn)

# TrTstSpecies = numberofChunks sized Species lists to use in training and testing. Includes all species except validation species
# ValSpecies = Validation species
# allSpecies = all species from data file. (Not unique, its size = number of rows in the data file)
allSpecies = Data[Data.columns[speciesColumn]]
#allSpecies = ["'" + s + "'" for s in allSpecies]

TrTstSpecies2=[]
for i in range(n_clusters):

    TrTstSpecies2.append(DMspecies[(clusters==i)])


first = 1 # to check if it is first training. Needed not to use hyperopt in each group. do not change
AllPR=[] # to keep all predictions
AllREAL=[] # to keep all real values related to predictions
Spec=[] # tested species list, relates to AllPR and AllREAL
random.shuffle(TrTstSpecies2)




TrTstSpecies=[]
cnt=0
for i in range(numberofChunks):
    tem=[]
    for iii in range(numberofClustersinEachChunk):
        temSpe=TrTstSpecies2[cnt+iii]
        for mic in temSpe:
            tem.append(mic)
    TrTstSpecies.append(tem)
    cnt=cnt+numberofClustersinEachChunk

TrTstSpecies = [[s.strip("'") for s in sublist] for sublist in TrTstSpecies]

for i in range(numberofChunks): # for each chunk
    trainSpecies=[]
    testSpecies = TrTstSpecies[i] # test species = ith group in the TrTstSpecies

    for j in range(numberofChunks): # This loop adds species to training set except ith group in the TrTstSpecies
        if j !=i:
            for asp in TrTstSpecies[j]:
                trainSpecies.append(asp)

    # Train the model and get predictions.
    myprediction, myrealValues = performCNN(maxtRNALen, optimizedParams, trainSpecies, testSpecies)

    # Store predictions' their real values and species names. This loop is not efficient but does not effect the performance
    # that much. Can be better.
    for l in range(len(myprediction)):
        AllPR.append(myprediction[l])
        AllREAL.append(myrealValues[l])
        Spec.append(testSpecies[l])
    # free keras memory
    gc.collect()
    K.clear_session()

#Calculate performance
AllPR=np.array(AllPR).ravel()
AllREAL=np.array(AllREAL).ravel()
mae, prsn, r2, rmse = measurePerformance(AllPR, AllREAL)
print(" Test performance mae= ", mae, " rmse= ", rmse, " pearsonR= ", prsn, " r2= ", r2)

# Write predictions, real values and species to file
writefile(AllPR, AllREAL,Spec, path)
#Write best params to console
print(optimizedParams)
