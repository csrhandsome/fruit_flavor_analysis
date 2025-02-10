from data_analysis.preprocessing import preprocess_data
from data_analysis.wgcna import wgcna
from model_runner.metabolomic_selection_runner import metabolomic_selection_runner
if __name__ == '__main__':
    preprocess_data(data_dir="data/supplemental_datasets/SD1_dataset_tomato.xlsx")
    wgcna(species="tomato", net_type="unsigned", ntraits=5, resdir="results/fig1/")
    metabolomic_selection_runner().run_metabolomic_selction()