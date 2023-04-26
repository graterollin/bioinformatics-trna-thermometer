import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def concatDf(dfList):
	df = dfList[0]
	for l in range(1, len(dfList)):
		df = pd.concat([df, dfList[l]], axis=0)
	return df

def readData(path, deli="\t", head=0):
	return pd.read_csv(path, header=head,delimiter = deli)

def elimData(dframe):
	trgt = dframe.values[:, 2] 

	cnt1 = (np.abs(trgt - 28.5) > 0.5)
	cnt2 = (np.abs(trgt - 30.5) > 0.5)
	cnt3 = (np.abs(trgt - 37.5) > 0.5)
	cnt = np.logical_and(cnt1, cnt2)
	cnt = np.logical_and(cnt, cnt3)

	res = dframe[cnt]
	return res.values

def prepareData(isBacteriaOn, isArchaeaOn, bacteria_file, archaea_file):
	if isBacteriaOn and isArchaeaOn:
		df_a = readData(archaea_file)
		#df_a = elimData(df_a)
		df_b = readData(bacteria_file)
		#df_b = elimData(df_b)
		df = concatDf([df_a, df_b])
	elif isBacteriaOn:
		df = readData(bacteria_file)
		#df = elimData(df)
	elif isArchaeaOn:
		df = readData(archaea_file)
		#df = elimData(df)

	return df

def main():
    # Input data
	isBacteriaOn = 1
	isArchaeaOn = 0
	
	bacteria_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Bacteria_RNAs_wOGT/Bacteria_RNAs_wOGT.txt"
	archaea_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Archaea_RNAs_wOGT/Archaea_RNAs_wOGT.txt"

	output_file = "D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/transformer-Output/output.txt"

	data = prepareData(isBacteriaOn, isArchaeaOn, bacteria_file, archaea_file)

	categorical_columns = ["GenomeAssembly", "Species", "OGT_Source", "tRNAid", "AminoAcid", "Anticodon", "AnticodonPosition", "Sequence", "Structure", "16s_rRNA", "23s_rRNA"]
	numerical_columns = ["Length", "CoveScore"]

	for col in categorical_columns:
		data[col] = data[col].astype("category").cat.codes

	train_data, test_data = train_test_split(data, test_size=0.2, random_state=20)


	# NOTE: IF THIS ERRORS OUT IT IS BECAUSE OF THE NEW ELIM DATA FUNC
	# I THINK THAT FUNCTION DOES THE SAME THING THAT THIS BLOCK OF CODE DOES
	X_train = train_data.drop(columns=["OGT"])
	y_train = train_data["OGT"]
	X_test = test_data.drop(columns=["OGT"])
	y_test = test_data["OGT"]

	# Standardize the numerical features
	scaler = StandardScaler()
	X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
	X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

	regressor = TabNetRegressor()
	regressor.fit(
		X_train.to_numpy(),
		y_train.to_numpy().reshape(-1, 1),
		eval_set=[(X_test.to_numpy(), y_test.to_numpy().reshape(-1, 1))],
	)

	# Test set prediction
	y_pred_test = regressor.predict(X_test.to_numpy())

	# Calculate the mean squared error
	mse = mean_squared_error(y_test, y_pred_test)
	print("Mean Squared Error: ", mse)

	mae = mean_absolute_error(y_test, y_pred_test)
	print("Mean Absolute Error: ", mae)
	
	rmse = np.sqrt(mse)
	print("Root Mean Squared Error: ", rmse)
	
	pearson_r, _ = pearsonr(y_test, y_pred_test.flatten())
	print("Pearson's R: ", pearson_r)
	
	r2 = r2_score(y_test, y_pred_test)
	print("R2: ", r2)

	output_data = pd.DataFrame({"Species": test_data["Species"],
                            "Real OGT Value": y_test,
                            "Predicted OGT Value": y_pred_test.flatten()})

	output_data.to_csv(output_file, index=False)

main()