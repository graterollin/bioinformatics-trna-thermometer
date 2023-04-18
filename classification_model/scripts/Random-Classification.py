from keras.models import model_from_json
import pandas as pd
import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout, concatenate
import tensorflow as tf
from keras.models import Model
import random
from hyperopt import tpe, hp, fmin
import gc
from keras.callbacks import EarlyStopping, History
from scipy.cluster.hierarchy import linkage,cut_tree
import copy
from matplotlib import pyplot

# Tensorflow Configirations
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.7
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

#config = tf.ConfigProto(allow_soft_placement=True)
#config.gpu_options.per_process_gpu_memory_fraction = 0.7
#sess = tf.Session(config=config)
#K.set_session(sess)


# We need some global variables. Because couldn't pass them to Hyperopt.

global leftTestX
global leftTrainX
global leftValidX
global rightTestX
global rightTrainX
global rightValidX
global ValidY
global TestY
global TrainY


def performCNN(leftTrainX, rightTrainX, maxtRNALen, args, YTrain, leftValidX, rightValidX,ValidY):
    # This function is a copy of objective function. It can take all parameters so we do not need global variables
    # The other difference is this function returns a prediction and real values not a loss.
    learning_rate = args['learning_rate']
    batch_size = args['batch_size']
    beta1 = args['beta1']
    beta2 = args['beta2']
    epochs = 30

    # create model
    mymodel = createModel( maxtRNALen, args)

    # Create early stopping (optional)
    es = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    hs = History()
    adam = keras.optimizers.Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2)

    mymodel.compile(loss="binary_crossentropy", optimizer=adam, metrics=['accuracy'])

    mymodel.fit([leftTrainX, rightTrainX], YTrain, batch_size=batch_size,
                epochs=epochs, verbose=1, validation_data=([leftValidX, rightValidX] ,ValidY), callbacks=[hs])

    return mymodel
def createModel(tRNALen, args):

    hidden_layer1=args['hidden_layer1']
    drop_out1 = args['drop_out1']
    filter_size1 = args['filter_size1']
    kernel_size1 = args['kernel_size1']
    pool_size1 = args['pool_size1']
    strides1 = args['strides1']

    convNumber = args['convNumber']
    denseLayerNumber = args['denseLayerNumber']



    # tRNA model left part input and output
    modelIntrnaLeft, modelOuttrnaLeft = buildModel(tRNALen, width=4, filterSize=filter_size1, kernelSize=kernel_size1,
                                         poolSize=pool_size1, denseSize=hidden_layer1, dropoutRate=drop_out1,
                                         strides=strides1,denseLayerNumber=denseLayerNumber,convNumber=convNumber)

    # tRNA model right part input and output
    modelIntrnaRight, modelOuttrnaRight = buildModel(tRNALen, width=4, filterSize=filter_size1, kernelSize=kernel_size1,
                                         poolSize=pool_size1, denseSize=hidden_layer1, dropoutRate=drop_out1,
                                         strides=strides1,denseLayerNumber=denseLayerNumber,convNumber=convNumber)


    # read tail parameters

    hidden_layer2=args['hidden_layer2']


    taildenseNumber = args['taildenseNumber']

    generalOut = buildCombine(modelOuttrnaLeft, modelOuttrnaRight, hidden_layer2,taildenseNumber)
    finalModel = Model(inputs=[modelIntrnaLeft, modelIntrnaRight], outputs=generalOut)

    return finalModel

def buildCombine(out1, out2, denseS, taildenseNumber):
    # Combines left and right branch of the CNN Model
    mymodel= concatenate([out1, out2])

    if taildenseNumber ==1:
        mymodel = Dense(denseS, activation='relu')(mymodel)
    if taildenseNumber ==2:
        mymodel = Dense(denseS, activation='relu')(mymodel)
        mymodel = Dense(denseS, activation='relu')(mymodel)
    if taildenseNumber ==3:
        mymodel = Dense(denseS, activation='relu')(mymodel)
        mymodel = Dense(denseS, activation='relu')(mymodel)
        mymodel = Dense(denseS, activation='relu')(mymodel)
    mOutput = Dense(2, activation='softmax')(mymodel)
    return mOutput

def buildModel(window_size, width=4, filterSize=0, kernelSize=0, poolSize=0,denseSize=0, dropoutRate=0, strides=0,denseLayerNumber=0,convNumber=0):
    # Builds a convolutional branch and returns input and output node
    #output node is not sized 1! It's an array.
    mInput = Input(shape=(width, window_size, 1))

    #Convolution
    if convNumber==1:
        model = Conv2D(filterSize, kernel_size=(4, kernelSize), padding='valid', activation='relu')(mInput)
        model = MaxPooling2D(pool_size=(4, poolSize),strides=(4, strides), padding='same')(model)
    elif convNumber==2:
        model = Conv2D(filterSize, kernel_size=(4, kernelSize), padding='valid', activation='relu')(mInput)
        model = MaxPooling2D(pool_size=(4, poolSize), strides=(4, strides), padding='same')(model)

        model = Conv2D(filterSize, kernel_size=(4, kernelSize), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(4, poolSize), strides=(4, strides),padding='same')(model)
    elif convNumber == 3:
        model = Conv2D(filterSize, kernel_size=(4, kernelSize), padding='valid', activation='relu')(mInput)
        model = MaxPooling2D(pool_size=(4, poolSize), strides=(4, strides), padding='same')(model)

        model = Conv2D(filterSize, kernel_size=(4, kernelSize), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(4, poolSize), strides=(4, strides), padding='same')(model)

        model = Conv2D(filterSize, kernel_size=(4, kernelSize), padding='same', activation='relu')(model)
        model = MaxPooling2D(pool_size=(4, poolSize), strides=(4, strides),padding='same')(model)

    # Dense layers
    model = Flatten()(model)
    if denseLayerNumber ==1:
        model = Dense(denseSize, activation='relu')(model)
        model = Dropout(dropoutRate)(model)
    elif denseLayerNumber ==2:
        model = Dense(denseSize, activation='relu')(model)
        model = Dropout(dropoutRate)(model)

        model = Dense(denseSize, activation='relu')(model)
        model = Dropout(dropoutRate)(model)

    elif denseLayerNumber == 3:
        model = Dense(denseSize, activation='relu')(model)
        model = Dropout(dropoutRate)(model)

        model = Dense(denseSize, activation='relu')(model)
        model = Dropout(dropoutRate)(model)

        model = Dense(denseSize, activation='relu')(model)
        model = Dropout(dropoutRate)(model)

    mOutput = Dense(denseSize, activation='linear')(model)

    return mInput, mOutput

def prepareData(isBacteriaOn, isArchaeaOn):
    # Takes two variables that indicates if we use archaea or bacteria
    # Returns read data as pandas data frame

    d = readData(path1)

    if isArchaeaOn and isBacteriaOn:
        dfA = readData(path1)
        dfB = readData(path2)
        d = concatDf([dfA, dfB])
    elif isArchaeaOn:
        d = readData(path1)
    elif isBacteriaOn:
        d = readData(path2)


    return d

def getTensor(mySeq, mxlen):
    # Takes data as pandas data frame
    # Returns selected column as tensor. Selected column should have RNA sequences. Also returns max lenght of the RNA
    # Convert SEQs data frame to CNN compatible one-hot encoded.
    SEQTensor = getOneHotEncodedSet(mySeq, mxlen=mxlen)
    return SEQTensor, mxlen


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

def getColumn(df, Col):
    # Takes a pandas data frame and returns a column
    res = df[df.columns[Col]]
    return pd.DataFrame(res.values)


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

def  getTarget(df,targetColumn):
    # Takes data frame and target column number
    # Returns target as array
    ogt = getColumn(df, Col=targetColumn).values
    ogt=np.array(ogt)
    OGT=[]
    # A faster may be added.
    for a in ogt:
        OGT.append(a[0])
    ogt= np.array(OGT)
    return ogt


def loadModel(path, modelname):
    # Loads a given model

    json_file = open(path+modelname+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path+modelname+".h5")
    print("Loaded model from disk")
    return loaded_model

def objective_func(args):
    # Objective function for hyper-parameter optimization.
    # Because the optimization type is minimization, we return 1-accuracy

    global leftTestX
    global leftTrainX
    global leftValidX
    global rightTestX
    global rightTrainX
    global rightValidX
    global ValidY
    global TestY
    global TrainY

    mymodel = performCNN(leftTrainX, rightTrainX, mxLen, args, TrainY, leftValidX, rightValidX,ValidY)
    preds = mymodel.predict([leftValidX, rightValidX])
    predLabel= np.argmax(preds,axis=1)
    acc=sum(predLabel==ValidY[:,1])/len(ValidY)

    K.clear_session()
    gc.collect()

    return 1-acc

def convertHyperopt(best_parameters):
    # Hyperopt retuns index of the selected hyper parameters.
    # This function takes indexes and returns best hyper parameters as dictionary.

    myspace = {'hidden_layer1': [4, 8, 16, 32, 64],
                      'drop_out1': [0, 0.05, 0.1, 0.2, 0.3],
                      'filter_size1': [4, 8, 16, 32],
                      'kernel_size1': [1, 3, 5, 8, 16],
                      'pool_size1': [3, 5, 8],
                      'strides1': [1, 3, 5, 8],
                      'hidden_layer2': [4, 8, 16, 32],
                      'batch_size': [16, 32, 64, 128],
                      'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                      'convNumber': [1, 2, 3],
                      'denseLayerNumber': [1, 2],
                      'taildenseNumber': [1, 2, 3]

                      }
    convertedDict={}
    for i in range(len(best_parameters)):

        itm = best_parameters.popitem()
        if itm[0]== 'beta1' or itm[0]== 'beta2':
            ky = itm[0]
            convertedDict[ky] = itm[1]
        else:
            ky=itm[0]
            vlind=itm[1]
            myarray = myspace[ky]
            myval=myarray[vlind]
            convertedDict[ky]=myval
    return convertedDict

def getrainTestValidationTSplits(Data, leftSpecsCol,rightSpecsCol, valRate, numberOfChunks):
    # Takes Data, random split on/off indicator, species column as number, validation rate and number of groups for train/test split\
    # If it is random split, first selects validation species randomly then divides the rest to number of chunks randomly
    # if it is distance split, first selects validation species randomly then divides the rest to number of chunks according to species distances

    leftSpecs = Data[Data.columns[leftSpecsCol]]
    rightSpecs = Data[Data.columns[rightSpecsCol]]
    mySpecies = np.concatenate([leftSpecs, rightSpecs])
    mySpecies = pd.Series(mySpecies)
    myUniqueSpecies = mySpecies.unique()

    random.shuffle(myUniqueSpecies)

    numberOfValSamples = int(valRate * len(myUniqueSpecies))
    ValSamples = myUniqueSpecies[:numberOfValSamples]
    TrTstSamples = myUniqueSpecies[numberOfValSamples:]
    TrTstSamples = np.array_split(TrTstSamples, numberOfChunks)

    return TrTstSamples, ValSamples, leftSpecs, rightSpecs

def getTrainTestValidControls(TrTstSpecies,fold,  leftSpecs, rightSpecs, ValSamples):
    # Prepares three filters for training, test and validation set.
    # Because we have pairs, we would like to have both species in the training set if a pair appears in training.

    TrSamples = []
    TstSamples = TrTstSpecies[fold]  # test species = ith group in the TrTstSpecies

    for j in range(numberofChunks):  # This loop adds species to training set except ith group in the TrTstSpecies
        if j != fold:
            for asp in TrTstSpecies[j]:
                TrSamples.append(asp)

    myRange = len(leftSpecs)
    isTrain = np.zeros(myRange)
    isTest = np.zeros(myRange)
    isValid = np.zeros(myRange)

    for i in range(myRange):
        if leftSpecs[i] in TrSamples and rightSpecs[i] in TrSamples:
            isTrain[i]=1
        if leftSpecs[i] in TstSamples and rightSpecs[i] in TstSamples:
            isTest[i]=1
        if leftSpecs[i] in ValSamples and rightSpecs[i] in ValSamples:
            isValid[i]=1

    return list(map(bool,isTrain)), list(map(bool,isTest)), list(map(bool,isValid))


def writefile(LS,VL, RS,VR, P, pth, fold):
    # P is predictions, R is real values, S is species array. Pth is path. fold is current chunk number
    for i in range(len(LS)):
        with open(pth+"Predictions-Random-" +str(fold)+".txt" , "a") as myfile:
            mystr = LS[i] +","+ str(VL[i]) + "," + RS[i] +","+ str(VR[i])+","+str(P[i]) + "\n"
            myfile.write(mystr)
    return
####################################   START     ##############################################################
####################################   INPUTS    ###############################################################




# Initializations

trnaSeqColumn=10 # in data file
speciesColumn = 1 # in data file

#Input files


#tRNAPath
path1 = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Archaea_tRNApairs_wOGT/Archaea_tRNApairs_wOGT.txt"
#path2 = ".../Bacteria_tRNApairs_wOGT_1Msample.txt"
path2 = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Bacteria_tRNApairs_wOGT_100Ksample/Bacteria_tRNApairs_wOGT_100Ksample.txt"
# provide path for output
#outputPath = ".../Desktop/"
outputPath = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/output.txt"

# hyper-parameter search space
myspace = {'hidden_layer1': hp.choice('hidden_layer1', [4, 8, 16, 32, 64]),
         'drop_out1': hp.choice('drop_out1', [0, 0.05, 0.1, 0.2, 0.3]),
         'filter_size1': hp.choice('filter_size1', [4, 8, 16, 32]),
         'kernel_size1': hp.choice('kernel_size1', [1, 3, 5, 8, 16]),
         'pool_size1': hp.choice('pool_size1', [3, 5, 8]),
         'strides1': hp.choice('strides1', [1, 3, 5, 8]),
         'hidden_layer2': hp.choice('hidden_layer2', [4, 8, 16, 32]),
         'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
         'convNumber': hp.choice('convNumber',[1, 2, 3]),
        'denseLayerNumber': hp.choice('denseLayerNumber',[1, 2]),
        'taildenseNumber': hp.choice('taildenseNumber',[1, 2, 3]),
         'beta1': hp.uniform('beta1', 0.9, 0.9999),
         'beta2': hp.uniform('beta2', 0.9, 0.9999),
         'learning_rate': hp.choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])}


####################################end of input parameters#############################################################
########################################################################################################################

min_ogt_diff=0 # Minimum OGT difference for training the model
maxtRNALen=0 # do not change

# One can change the following parameters ragarding to the input file format

#Column IDs
lefttRNAcolumn =4
righttRNAcolumn =9

leftTarget=1
rightTarget=6

leftSpecCol=0
rightSpecCol=5

max_eval = 30 # Max iteration for hyper-parameter optimization
ValRate=0.05 # Validation set percentage
numberofChunks = 5 # To divide data set into groups after validation set is excluded.


ishyperOptNeeded=1 # Hyper-Parameter on-off
isArchaeaOn=1 # Archaea on-off
isBacteriaOn=0 # Bacteria on-off

#Change parameters if there is a new optimized parameters
if ishyperOptNeeded == 0:  # if hyper parameter optimization will not work
    if isArchaeaOn == 1 and isBacteriaOn == 0:  # only Archaea
        optimizedParams = {'taildenseNumber': 1, 'strides1': 1, 'pool_size1': 8,
                           'learning_rate': 0.0001, 'kernel_size1': 8, 'hidden_layer2': 32,
                           'hidden_layer1': 64, 'filter_size1': 16, 'drop_out1': 0, 'denseLayerNumber': 1,
                           'convNumber': 1, 'beta2': 0.9679243807462391, 'beta1': 0.974719394657258, 'batch_size': 128} # will be replaced with best parameters
    elif isArchaeaOn == 0 and isBacteriaOn == 1:  # only Bacteria
        optimizedParams ={'taildenseNumber': 2, 'strides1': 1, 'pool_size1': 3, 'learning_rate': 0.0001, 'kernel_size1': 16,
                          'hidden_layer2': 16, 'hidden_layer1': 64, 'filter_size1': 16, 'drop_out1': 0.1, 'denseLayerNumber': 2,
                          'convNumber': 2, 'beta2': 0.9121970671679941, 'beta1': 0.9093874847896459, 'batch_size': 128} # will be replaced with best parameters
    else:  # Archaea and Bacteria together
        optimizedParams = {'taildenseNumber': 2, 'strides1': 3, 'pool_size1': 3, 'learning_rate': 0.0001, 'kernel_size1': 16, 'hidden_layer2': 4, 'hidden_layer1': 16,
                           'filter_size1': 8, 'drop_out1': 0, 'denseLayerNumber': 1, 'convNumber': 1, 'beta2': 0.963857681231603, 'beta1': 0.9680013160667001, 'batch_size': 16}


# Read data from files
DataRaw = prepareData(isBacteriaOn, isArchaeaOn)


# Data preparation
LeftYTarget = getTarget(DataRaw, leftTarget)
RigtYTarget = getTarget(DataRaw, rightTarget)
filterCont=(np.abs(LeftYTarget - RigtYTarget)>min_ogt_diff) #Exclude tRNA pairs if OGT difference lower than the given threshold
Data=pd.DataFrame(DataRaw[filterCont].values)

LSEQs = getColumn(Data, Col=lefttRNAcolumn)
mxlenLeft = len(max(LSEQs.values[:, 0], key=len))
RSEQs = getColumn(Data, Col=righttRNAcolumn)
mxlenRight = len(max(RSEQs.values[:, 0], key=len))

mxLen=max(mxlenRight,mxlenLeft)


LefttRNA, leftLen = getTensor(LSEQs,mxLen)
RighttRNA, rightLen = getTensor(RSEQs,mxLen)

LeftYTarget = getTarget(Data, leftTarget)
RigtYTarget = getTarget(Data, rightTarget)

# Target is 2-d vector: 01 or 10 according to left side is greater or otherwise.
Target = np.zeros((len(LeftYTarget),2))
Target[(LeftYTarget > RigtYTarget),0] = 1
Target[(LeftYTarget < RigtYTarget),1] = 1

#Splitting training, test and validation species.
TrTstSpecies, ValSpecies, leftSpecs, rightSpecs = getrainTestValidationTSplits(Data, leftSpecCol,rightSpecCol, ValRate, numberofChunks)
first = 1  # to check if it is first training. Needed not to use hyperopt in each group. do not change

#Run the model and get predictions "numberofChunks" times. We do 5.
for fold in range(numberofChunks):

    isTrain, isTest, isValid= getTrainTestValidControls(TrTstSpecies,fold,  leftSpecs, rightSpecs, ValSpecies)

    # Get training data
    leftTrainX = LefttRNA[isTrain,:,:,:]
    rightTrainX = RighttRNA[isTrain,:,:,:]

    leftValidX =LefttRNA[isValid,:,:,:]
    rightValidX = RighttRNA[isValid,:,:,:]


    leftTestX =LefttRNA[isTest,:,:,:]
    rightTestX = RighttRNA[isTest,:,:,:]

    TrainY=Target[isTrain,:]
    TestY = Target[isTest,:]
    ValidY = Target[isValid,:]

    #############################################
    #############################################

    AllPR=[] # to keep all predictions
    AllREAL=[] # to keep all real values related to predictions
    Spec=[] # tested species list, relates to AllPR and AllREAL


    if first==1 and ishyperOptNeeded==1: # if it is first run then best hyper parameters needed to be found. Call hyperoptimization
        best_parameters = fmin(objective_func, myspace,
                          algo=tpe.suggest, max_evals=max_eval)
        convertedParams=convertHyperopt(best_parameters) # best_parameters are indices. Convert appropiate format.
        f = open(outputPath + "best_Parameters.txt", "w") #keep best parameters
        f.write(str(convertedParams))
        f.close()

        first=0 # First run is finished. We get best parameters. And, do not want to do heyperopt in the next iteration.
    elif ishyperOptNeeded==0:
        convertedParams = optimizedParams

#############################################
#############################################

    PR=[] # to keep all predictions
    Spec=[] # tested species list, relates to AllPR and AllREAL

    # Get the model
    print(f"Shape of leftTrainX: {leftTrainX.shape}")
    print(f"Shape of rightTrainX: {rightTrainX.shape}")
    mymodel = performCNN(leftTrainX, rightTrainX, mxLen, convertedParams, TrainY, leftValidX, rightValidX,ValidY)
    # Predict test set.
    preds = mymodel.predict([leftTestX, rightTestX])
    predLabel= np.argmax(preds,axis=1)

    # Write predictions to a txt file
    writefile(np.array(leftSpecs[isTest]), np.array(LeftYTarget[isTest]), np.array(rightSpecs[isTest]), np.array(RigtYTarget[isTest]), predLabel, outputPath, fold)


    gc.collect()
    K.clear_session()