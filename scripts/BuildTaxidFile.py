# Andres Graterol 
# CAP 6545
# tRNA thermometer project
# ================================================
# Builds the taxid and species file from the file given by ncbi
# Formats file in accordance to what is needed to create the distance matrix from Tree
# ================================================
import pandas as pd

def process_file(input_file, output_file):
    df = pd.read_csv(input_file, delimiter='\t|\t', engine='python', usecols=['name', 'taxid'])

    # rearrange the dataframe columns to have the taxid go first 
    cols = df.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    # Rename the columns 
    df['1'] = df['taxid']
    df['Species'] = df['name']

    # header matches that of 5_getTaxids.py
    header = ['1', 'Species']
    df.to_csv(output_file, sep='\t', index=False, columns=header)
    return 

def main():
    # File Paths 
    archaea_file = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/ArchaeaAndTaxid.txt'
    bacteria_file = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/BacteriaAndTaxid.txt'
    combined_file = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/ArchaeaAndBacteriaTaxid.txt'

    # Output Paths 
    output_archaea = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/FormattedSpeciesAndTaxid/FormattedArchaea.txt'
    output_bacteria = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/FormattedSpeciesAndTaxid/FormattedBacteria.txt'
    output_both = 'D:/Grad_School/Spring_23/BioInformatics/final_project/MyResults/data/input_data/SpeciesAndTaxId/FormattedSpeciesAndTaxid/FormattedArchaeaAndBacteria.txt'

    # Process the files
    process_file(archaea_file, output_archaea)
    process_file(bacteria_file, output_bacteria)
    process_file(combined_file, output_both)

    return 

main()