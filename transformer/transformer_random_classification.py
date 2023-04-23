import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

def concatDf(dfList):
	df = dfList[0]
	for l in range(1, len(dfList)):
		df = pd.concat([df, dfList[l]], axis=0)
	return pd.DataFrame(df.values)

def readData(path, deli="\t", head=0):
	return pd.read_csv(path, header=head,delimiter = deli)

def prepareData(isBacteriaOn, isArchaeaOn, bacteria_file, archaea_file):
	if isBacteriaOn and isArchaeaOn:
		df_a = readData(archaea_file)
		df_b = readData(bacteria_file)
		df = concatDf([df_a, df_b])
	elif isBacteriaOn:
		df = readData(bacteria_file)
	elif isArchaeaOn:
		df = readData(archaea_file)

	return df


def main():
	# Input data
	isBacteriaOn = 0
	isArchaeaOn = 1
	
	bacteria_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Bacteria_tRNApairs_wOGT_100Ksample/Bacteria_tRNApairs_wOGT_100Ksample.txt"
	archaea_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Archaea_tRNApairs_wOGT/Archaea_tRNApairs_wOGT.txt"

	prepareData(isBacteriaOn, isArchaeaOn, bacteria_file, archaea_file)



main()