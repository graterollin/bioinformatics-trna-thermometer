from random import random
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

def preform_cv(X_train_val, y_train_val):
	# Set up 5-fold CV 
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
	# Used to store the validation accuracies on each fold of CV
	val_accuracies = []

	for train_index, val_index in kfold.split(X_train_val, y_train_val):
		X_train, X_val = X_train_val[train_index], X_train_val[val_index]
		y_train, y_val = y_train_val[train_index], y_train_val[val_index] 

		# Train TabNet 
		clf = TabNetClassifier(seed=20)
		clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

		# Validation set prediction
		y_pred_val = clf.predict(X_val)

		# Compute validation accuracy 
		val_accuracy = accuracy_score(y_val, y_pred_val)
		val_accuracies.append(val_accuracy)

	# Calculate the mean and SD for the 5 chunks 
	mean_val_accuracy = np.mean(val_accuracies)
	print("Mean Val Accuracy: ", mean_val_accuracy)

	standard_deviation = np.std(val_accuracies)
	print("Standard Deviation: ", standard_deviation)

def concatDf(dfList):
	df = dfList[0]
	for l in range(1, len(dfList)):
		df = pd.concat([df, dfList[l]], axis=0)
	return df

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
	isBacteriaOn = 1
	isArchaeaOn = 1
	
	bacteria_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Bacteria_tRNApairs_wOGT_100Ksample/Bacteria_tRNApairs_wOGT_100Ksample.txt"
	archaea_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Archaea_tRNApairs_wOGT/Archaea_tRNApairs_wOGT.txt"

	output_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/transformer-Output/output.txt"

	data = prepareData(isBacteriaOn, isArchaeaOn, bacteria_file, archaea_file)

	categorical_columns = ['Species1', 'tRNAid1', 'Anticodon1', 'tRNA_Sequence1', 'Species2', 'tRNAid2', 'Anticodon2', 'tRNA_Sequence2']
	numerical_columns = ['OGT1', 'OGT2']

	data_encoded = data.copy()
	for col in categorical_columns:
		le = LabelEncoder()
		data_encoded[col] = le.fit_transform(data[col])

	X = data_encoded.drop(columns=['OGT1', 'OGT2'])
	y = (data_encoded['OGT1'] > data_encoded['OGT2']).astype(int).values

	X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=20, stratify=y)

	test_indices = X_test.index

	X_train_val = X_train_val.to_numpy()
	X_test = X_test.to_numpy()

	# Standardize the OGT columns between 0 and 1
	scaler = StandardScaler()
	X_train_val[:, -len(numerical_columns):] = scaler.fit_transform(X_train_val[:, -len(numerical_columns):])
	X_test[:, -len(numerical_columns):] = scaler.transform(X_test[:, -len(numerical_columns):])

	# Comment/uncomment as needed to use CV 
	#preform_cv(X_train_val, y_train_val)

	# Train TabNet on the combined train and val set 
	clf = TabNetClassifier(seed=20)
	clf.fit(X_train_val, y_train_val)

	# Testing predictions
	y_pred_test = clf.predict(X_test)

	# Compute testing accuracy 
	test_accuracy = accuracy_score(y_test, y_pred_test)
	print("Testing Accuracy: ", test_accuracy)

	# Output the final CSV file 
	test_data = data.iloc[test_indices]

	predictions = pd.DataFrame({'Species1': test_data['Species1'],
                                 'OGT1': test_data['OGT1'],
                                 'Species2': test_data['Species2'],
                                 'OGT2': test_data['OGT2'],
                                 'Prediction_label': y_pred_test})

	predictions.to_csv(output_file, index=False)

main()