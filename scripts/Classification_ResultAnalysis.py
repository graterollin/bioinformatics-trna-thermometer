# Andres Graterol 
# CAP 6545
# tRNA thermometer project
# =========================
import pandas as pd
import numpy as np

def colAsArray(file, column):
    col = file[file.columns[column]]
    col = pd.DataFrame(col.values).values
    col = np.array(col)
    
    COL=[]
    for a in col:
        COL.append(a[0])
    col = np.array(COL)

    return col

def predictionChecker(temp1, temp2, guess):
    if ((temp1 > temp2) and guess == 0):
        return 1
    elif ((temp2 > temp1) and guess == 1):
        return 1
    
    return 0

def computeAccuracy(ogt1, ogt2, prediction):
    # Bins to hold each range 
    occuranceBins = {
        '0and5': 0,
        '5and10': 0,
        '10and20': 0,
        '20and30': 0,
        '30andAbove': 0
    }

    correctPredictionBins = {
        '0and5': 0,
        '5and10': 0,
        '10and20': 0,
        '20and30': 0,
        '30andAbove': 0
    }

    accuracyBins = {
        '0and5': 0.0,
        '5and10': 0.0,
        '10and20': 0.0,
        '20and30': 0.0,
        '30andAbove': 0.0
    }

    numberOfRows = len(prediction)
    row = 0
    for x in range(numberOfRows):
        if (0<=abs(ogt1[row]-ogt2[row])<5):
            # Add it to the running count
            occuranceBins['0and5'] += 1
            # Update the correct predictions counter
            correctPredictionBins['0and5'] += predictionChecker(ogt1[row], ogt2[row], prediction[row])
        elif (5<=abs(ogt1[row]-ogt2[row])<10):
            occuranceBins['5and10'] += 1
            correctPredictionBins['5and10'] += predictionChecker(ogt1[row], ogt2[row], prediction[row])
        elif (10<=abs(ogt1[row]-ogt2[row])<20):
            occuranceBins['10and20'] += 1
            correctPredictionBins['10and20'] += predictionChecker(ogt1[row], ogt2[row], prediction[row])
        elif (20<=abs(ogt1[row]-ogt2[row])<30):
            occuranceBins['20and30'] += 1
            correctPredictionBins['20and30'] += predictionChecker(ogt1[row], ogt2[row], prediction[row])
        elif (30<=abs(ogt1[row]-ogt2[row])):
            occuranceBins['30andAbove'] += 1
            correctPredictionBins['30andAbove'] += predictionChecker(ogt1[row], ogt2[row], prediction[row])
        else:
            raise Exception("Invalid criteria reached!")
        
        row += 1

    print("correct prediction bins:", correctPredictionBins)
    print("occurance bins:", occuranceBins)

    # Compute accuracy for each grouping
    accuracyBins['0and5'] = correctPredictionBins['0and5'] / occuranceBins['0and5']
    accuracyBins['5and10'] = correctPredictionBins['5and10'] / occuranceBins['5and10']
    accuracyBins['10and20'] = correctPredictionBins['10and20'] / occuranceBins['10and20']
    accuracyBins['20and30'] = correctPredictionBins['20and30'] / occuranceBins['20and30']
    accuracyBins['30andAbove'] = correctPredictionBins['30andAbove'] / occuranceBins['30andAbove']

    return accuracyBins

def analyzeFile(filepath):
    # Column numbers 
    tRNA1 = 1
    tRNA2 = 3
    result = 4

    file = pd.read_csv(filepath, header=None, delimiter=",")
    ogt1 = colAsArray(file, tRNA1)
    ogt2 = colAsArray(file, tRNA2)
    prediction = colAsArray(file, result)

    accuracies = computeAccuracy(ogt1, ogt2, prediction)

    return accuracies

def main():
    # File paths for the three files
    isRandomSplit = True

    # Choose which file you wish to evaluate
    isArchaea = True 
    isBacteria = True 

    if (isRandomSplit):
        # Random Split 
        archaeaFilePath = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/RandomSplit/ArchaeaOutput/ArchaeaPredictions-Random.txt'
        bacteriaFilePath = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/RandomSplit/BacteriaOutput/BacteriaPredictions-Random.txt'
        archaeaAndBacteriaFilePath = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/RandomSplit/ArchaeaAndBacteriaOutput/ArchaeaAndBacteriaPredictions-Random.txt'
    else:
        # Phylogenetic Split
        archaeaFilePath = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/PhylogeneticSplit/ArchaeaOutput/ArchaeaPredictions-Random.txt'
        bacteriaFilePath = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/PhylogeneticSplit/BacteriaOutput/BacteriaPredictions-Random.txt'
        archaeaAndBacteriaFilePath = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/classification_model-Output/PhylogeneticSplit/ArchaeaAndBacteriaOutput/ArchaeaAndBacteriaPredictions-Random.txt'

    # Analyze the file
    if (isArchaea and isBacteria):
        results = analyzeFile(archaeaAndBacteriaFilePath)
    elif (isArchaea):
        results = analyzeFile(archaeaFilePath)
    elif (isBacteria):
        results = analyzeFile(bacteriaFilePath)
    else:
        raise Exception("No file type specified")

    print(results)

main()