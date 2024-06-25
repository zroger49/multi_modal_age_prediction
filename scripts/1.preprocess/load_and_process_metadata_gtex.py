import pandas as pd


def load_and_process_metadata(metadata_tab_file: str, index_col:str ="SAMPIDDOT") -> pd.DataFrame:
    metadata = pd.read_csv(metadata_tab_file,
                           sep="\t", header=0, index_col=index_col)
    metadata = metadata.drop(metadata.columns[0], axis=1)  # remove unnecessary column in our metadata
    # replace NaN values with 0
    metadata = metadata.fillna(0.0)
    # create new column with tissue sample id
    metadata.insert(value=[".".join(idx.split(".")[0:3]).replace(".", "-") for idx in metadata.index],
                    column="tissue_sample_id",
                    loc=0)
    # add age bin information with bins of size 5
    metadata.insert(value=pd.cut(metadata["AGE"], labels=False,
                                 bins=list(range(int(metadata["AGE"].min())-1, 
                                                 int(metadata["AGE"].max())+5+1, 5))),
                    column="age_bins",
                    loc=2)
    # change index separators from dots to dashes for consistency
    metadata.index = metadata.index.str.replace(".", "-", regex=False)
    # rename index to make more sense
    metadata = metadata.rename_axis("rnaseq_sample_ids")
    # select metadata of interest
    metadata = metadata.loc[:, [ "tissue_sample_id", "AGE", "age_bins", "GENDER", "HGHT", "WGHT", "BMI", 
                                "SMTSD", "SMTSISCH", "DTHHRDY"]]
    # rename to more understandable and consistent names
    metadata = metadata.rename({"AGE": "age", "GENDER": "gender", "HGHT": "height", "WGHT": "weight",
                                "DTHHRDY": "hardy_scale", "BMI": "bmi", "SMTSD": "tissue", "SMTSISCH": "ischemic_time"},
                                axis=1)
    metadata = metadata.reset_index()
    metadata = metadata.set_index("tissue_sample_id")
    return metadata


if __name__ == '__main__':
    metadata = load_and_process_metadata("../data/metadata/gtex_v8_metadata_full.tab")
    print(metadata[metadata["tissue"].str.lower() == "lung"].shape)
