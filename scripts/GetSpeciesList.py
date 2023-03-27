# Andres Graterol 
# CAP 6545
# tRNA thermometer project
# ================================================
# Builds the taxid and species corresponding list  
# from the input RNA files
# ================================================
import pandas as pd
import numpy as np

def mergeFiles(dfList):
    # Concats list of data frames into one data frame
    df = dfList[0]
    for l in range(1, len(dfList)):
        df = pd.concat([df, dfList[l]], axis=0)

    header = ['Species']
    return pd.DataFrame(df.values, columns=header)

def process_file(input_file, output_file):
    df = pd.read_csv(input_file, delimiter='\t', usecols=['Species'])
    
    # Replace the underscore in the species name with a whitespace 
    df['Species'] = df['Species'].str.replace('_', ' ')
    # Remove duplicate species
    df = df.Species.unique()
    print("Unique species in dataframe: ", len(df))
    
    header = ['Species']
    df = pd.DataFrame(df, columns=header)
    #print(df)
    
    # Uncomment the line below if want result outputted to the file 
    #df.to_csv(output_file, index=False)

    return df

def main():
    # NOTE: Should get 165 unique Archaea (yes)
    #       And get 2375 unique Bacteria (getting 2376, paper eliminates one that is unsuitable)

    # NOTE: Try using the input files associated with the RNA first
    #       These are the files associated with the regression model
    # TODO: Check that all tRNA species are included in the RNA file as well 
    #       so no issues arise when we run the classification model

    # Archaea File first 
    input_file1 = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Archaea_RNAs_wOGT/Archaea_RNAs_wOGT.txt'
    # Bacteria File second
    input_file2 = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/Bacteria_RNAs_wOGT/Bacteria_RNAs_wOGT.txt'

    # Analyze the separate files before we combine them into one file
    output_file1 = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesLists/ArchaeaRNASpeciesList.txt'
    df1 = process_file(input_file1, output_file1)
    output_file2 = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesLists/BacteriaRNASpeciesList.txt'
    df2 = process_file(input_file2, output_file2)

    # Combine the two files 
    output_merged_file = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesLists/ArchaeaAndBacteriaSpeciesList.txt'
    df = mergeFiles([df1, df2])
    print(df)
    df.to_csv(output_merged_file, index=False)

    return 

main()