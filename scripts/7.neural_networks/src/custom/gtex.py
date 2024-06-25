import os
from histolab.slide import Slide
from pathlib import Path
import pandas as pd
import numpy as np
from ..gen_patch.create_patches_fp import patch_one_wsi
import openslide
import yaml
import torch
from tqdm import tqdm
#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#print("PYTORCH_ENABLE_MPS_FALLBACK: ", os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

import yaml
def get_parameters():
    with open("./conf/params__global.yml") as params:
        params_dict = yaml.safe_load(params)
    return params_dict

params = get_parameters()

path__gtex_portal_data = params["PATH__GTExPortalDB"]
path__smoker_annotation = params["PATH__SmokerStatus"]
path__donor_age = params["PATH__DonorAge"]

path__methylation = params['PATH__Methylation']
path__histology = params['PATH__Histology']
path__gene = params['PATH__GeneExpression']

DEVICE = params["DEVICE"]
#path__test_sample = params["PATH_TestSample"].format(params["TISSUE_TYPE"])

# =====================
# INITIALIZE HIPT
# =====================
from functools import lru_cache
# from src.HIPT_4K.hipt_4k import HIPT_4K
# from src.HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
# from src.HIPT_4K.hipt_heatmap_utils import *

# light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)

# pretrained_weights256 = './src/HIPT_4K/Checkpoints/vit256_small_dino.pth'
# pretrained_weights4k = './src/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'
# device256 = torch.device('mps')
# device4k = torch.device('mps')

# ### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
# model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
# model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

# ### ViT_256 + ViT_4K loaded into HIPT_4K API
# model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
# model.eval()

from src.HIPT_4K.hipt_4k import HIPT_4K
from src.HIPT_4K.hipt_model_utils import eval_transforms
@lru_cache(maxsize=None)  # Replace with @cache() if we don't want to limit cache size
def get_hipt_model():
    pretrained_weights256 = './src/HIPT_4K/Checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = './src/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'
    device256 = torch.device(DEVICE)
    device4k = torch.device(DEVICE)

    model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
    model.eval()

    return model
# =====================

def list_non_empty_dir_names(directory_path):
    """
    Returns a list of non-empty directory names, where each directory
    contains other folders or files, for the given directory path using pathlib.
    Ignores directories named 'embeddings'.
    """
    path = Path(directory_path)
    # non_empty_dir_names = [
    #     item.name for item in path.iterdir() if item.is_dir() and item.name != "embeddings" and any(item.iterdir())
    # ]
    
    dir_names_with_specific_file = [
        item.name for item in path.iterdir() 
        if item.is_dir() 
        and (item / "embeddings" / "asset_dict.pth").exists()
    ]
    return dir_names_with_specific_file

dfpd_GtexPortal = pd.read_csv(path__gtex_portal_data).rename(
        {
            "Tissue Sample ID":"sample_id", 
            "Subject ID":"subject_id"
        }, axis=1
    )

dfpd_SmokerStatus = pd.read_csv(path__smoker_annotation, sep=";").rename(
        {
            "Donor":"subject_id", 
            "SmokerStatus":"smoker_status"
        }, axis=1
    )

dfpd_DonorAge = pd.read_csv(path__donor_age).rename(
        {
            "Individual_ID":"subject_id", 
            "Age":"age"
        }, axis=1
    )

def get_pathology_array(sid, tissue_type, dfpd_GtexPortal=dfpd_GtexPortal):
    dfpd_pathologies = dfpd_GtexPortal.query(f"Tissue == '{tissue_type}'")["Pathology Categories"].str.split(", ").explode()
    dfpd_pathologies = pd.get_dummies(dfpd_pathologies).groupby(level=0).sum()
    dfpd_pathologies = dfpd_pathologies.set_index(dfpd_GtexPortal.loc[dfpd_pathologies.index]["sample_id"])
    return dfpd_pathologies.loc[sid].to_dict()

def get_smoker_status(subject_id):
    dfpd_temp = dfpd_SmokerStatus[dfpd_SmokerStatus["subject_id"] == subject_id]
    if dfpd_temp.empty:
        return "unknown"
    elif dfpd_temp.MHSMKPRD.to_list()[0] in ['Non Smoker']:
        return "Non Smoker"
    elif dfpd_temp.MHSMKCMT.to_list()[0] in ['Non Smoker']:
        return "Smoker"
    else:
        smoker_status = dfpd_temp.smoker_status.fillna("unknown").to_list()[0]
        return smoker_status

def get_age(subject_id):
    try:
        dfpd_temp = dfpd_DonorAge[dfpd_DonorAge["subject_id"] == subject_id]
        age = dfpd_temp["age"].to_list()[0]
        return age
    except:
        return np.nan

class GtexSample():
    def __init__(self, path__gtex_portal_data=path__gtex_portal_data):
        self.database = pd.read_csv(path__gtex_portal_data).rename(
                {
                    "Tissue Sample ID":"sample_id", 
                    "Subject ID":"subject_id"
                }, axis=1
            )
        
        self.attributes = {
            "tissues": list(self.database["Tissue"].unique())
        }

    def get_subsample(self, key, value):
        df_temp = self.database[self.database[key] == value]
        lt_sid = list(df_temp.sample_id.unique())

        print(f"[+] Found {len(lt_sid)} subjects for {value}")
        return list(df_temp.sample_id.unique())

    def compare_subsamples(self, key, values, return_subsamples=False):
        dct_sets = {t:set(["-".join(i.split("-")[:-1]) for i in self.get_subsample(key, t)]) for t in values}

        import matplotlib.pyplot as plt
        import venn

        venn_diagram = venn.venn(dct_sets, cmap="plasma", fontsize=8, figsize=(5,5))
        plt.title(f'Number of subjects per {key}')
        plt.show()

        if return_subsamples:
            return dct_sets

class GtexSlide():
    def __init__(self, sid):
        global path__histology
        self.sid = sid
        self.subject_id = "-".join(sid.split('-')[:-1])
        #self.output = os.path.join(path__histology, self.tissue)
        self.attributes = {
            "GTExPortal": dfpd_GtexPortal[dfpd_GtexPortal["sample_id"] == self.sid].to_dict("records")[0]
        }

        self.tissue = self.attributes["GTExPortal"]["Tissue"]

        self.output = os.path.join(path__histology, self.tissue)
        self.tile_path = f'{self.output}/{self.sid}'
        self.slide_path = f'{self.output}/{self.sid}.svs'
        # self.attributes.update(
        #     {"pathologies" : get_pathology_array(self.sid, self.tissue)}
        #     )

        # self.attributes.update(
        #     {"smoker_status": get_smoker_status(self.subject_id)}
        # )

        # self.attributes.update(
        #     {"age": get_age(self.subject_id)}
        # )

        self.hipt_features =  ["features_mean256_cls4k", "features_mean256", "features_cls4k"]

        if not os.path.exists(self.tile_path):
            Path(self.tile_path).mkdir(parents=True, exist_ok=True)
            print("[+] Tile path created.")
        if not os.path.exists(self.output):
            Path(self.output).mkdir(parents=True, exist_ok=True)
            print("[+] Output path created.")

    def delete_wsi(self):
        rm_svs = f"rm {self.slide_path}"
        os.system(rm_svs)
        print(f"{self.sid} removed")
        print()
    
    def get_wsi(self):
        # Check if the file already exists
        if os.path.exists(self.slide_path):
            print(f"[-] File {self.sid} already exists.")
            self.slide = openslide.OpenSlide(self.slide_path)
            return False
        else:
            try:
                print(f"[+] Downloading wsi {self.sid}...")
                curl_cmd = f"curl 'https://brd.nci.nih.gov/brd/imagedownload/{self.sid}' --output '{self.slide_path}'"
                os.system(curl_cmd)
                self.slide = openslide.OpenSlide(self.slide_path)
                return True
            except Exception as e:
                print(f'Problems downloading wsi {self.sid}: {e}')
    
    def info(self):
        print(f"Slide name: {self.sid}")

        # Print basic information about the slide
        print("Slide Dimensions: ", self.slide.dimensions)
        print("Slide Level Count: ", self.slide.level_count)
        print("Slide Level Dimensions: ", self.slide.level_dimensions)
        print("Slide Level Downsamples: ", self.slide.level_downsamples)

        # Print all available properties/metadata
        print("\nSlide Properties (Metadata):")
        for key, value in self.slide.properties.items():
            print(f"{key}: {value}")

        # Optionally, close the slide after processing
        #slide.close()

    def break_into_tiles(self, patch_size=4096, step_size=4096):
        patch_one_wsi(
            wsi_full_path=self.slide_path, 
            save_dir=self.tile_path, 
            patch_size=patch_size, 
            step_size=step_size
            )
        
    def get_hipt_embeddings(self):
        """
        Depends of break_into_tiles.
        """
        self.patch_path = os.path.join(self.tile_path, f'patches/{self.sid}.h5')
        import h5py
        coords = h5py.File(self.patch_path, 'r')['coords']

        df_coords = pd.DataFrame(list(coords), columns=["x", "y"])
        df_coords["white_bg"] = np.nan
        df_coords["mean"] = np.nan
        df_coords["std"] = np.nan
        df_coords["cv"] = np.nan
        df_coords["approved"] = np.nan

        size = 4096
        wsi_asset_dicts = {}

        for index, row in tqdm(df_coords.iterrows(), total=df_coords.shape[0]):
            x = int(row['x'])
            y = int(row['y'])

            region = self.slide.read_region((x,y), 0, (size, size)).convert('RGB')

            # Save thumbnail
            thumb = region.copy()
            thumb.thumbnail((128,128))
            thumb_path = f"{self.tile_path}/thumbnails"
            region_name = f"{self.sid}_tile_{size}_{index}"

            if not os.path.exists(thumb_path):
                Path(thumb_path).mkdir(parents=True, exist_ok=True)
                print("[+] Thumb path created.")
            
            thumb.save(os.path.join(thumb_path, region_name) + ".png")
            
            # Calculate metrics
            ar_reg = np.array(region)

            white_bg = np.sum(ar_reg > 230)/(ar_reg.shape[0]*ar_reg.shape[1]*ar_reg.shape[2])
            std_val = np.std(ar_reg)
            mean_val = np.mean(ar_reg)
            cv_val = std_val / mean_val
            
            # Update the DataFrame directly
            df_coords.at[index, "white_bg"] = white_bg
            df_coords.at[index, "std"] = std_val
            df_coords.at[index, "mean"] = mean_val
            df_coords.at[index, "cv"] = cv_val
            
            # Condition to display the image
            if white_bg > 0.5 and cv_val <= 0.35:
                df_coords.at[index, "approved"] = 0
            else:
                df_coords.at[index, "approved"] = 1

            # Get embeddings from HIPT for every tile
            x = eval_transforms()(region).unsqueeze(dim=0)
            model = get_hipt_model()
            asset_dict = model.forward_asset_dict(x)

            wsi_asset_dicts[region_name] = asset_dict
        
        df_coords.to_csv(os.path.join(thumb_path, "region_score.csv"))
        print("[+] Coordinates dataframe saved with scores.")

        embeddings_path = f"{self.tile_path}/embeddings"
        if not os.path.exists(embeddings_path):
            Path(embeddings_path).mkdir(parents=True, exist_ok=True)
            print("[+] Embeddings path created.")

        torch.save(wsi_asset_dicts, os.path.join(embeddings_path, "asset_dict.pth"))
        print("[+] Embeddings dict saved.")
    
    def get_hipt_features(self, feature=None):
        if feature:
            hipt_features = feature
        else:
            hipt_features =  self.hipt_features
        embeddings_path = f"{self.tile_path}/embeddings"

        rep_dict = {}
        for feature in hipt_features:
            lt_wsi = []
            try:
                p = os.path.join(embeddings_path, 'asset_dict.pth')
                asset_dict = torch.load(p)
                patch_list = []
                for patch in asset_dict.keys():
                    tensor_current = asset_dict[patch][feature]
                    patch_list.append(tensor_current)
                lt_wsi.append(torch.cat(patch_list, dim=0).mean(dim=0).unsqueeze(0))
            except:
                print(f'[-] Problems opening {self.sid} folder.')
            rep_dict[feature] = torch.cat(lt_wsi, dim=0)
        return rep_dict

    def get_tiles_path(self, prefix_path=None):
        if prefix_path:
            self.prefix_path = f"{self.tile_path}/{prefix_path}"
        
        path = f"{self.prefix_path}"
        return os.listdir(path)

class GtexMethylation():
    def __init__(self, sid):
        global path__methylation
        self.sid = sid

        self.attributes = {
            "GTExPortal": dfpd_GtexPortal[dfpd_GtexPortal["sample_id"] == self.sid].to_dict("records")[0]
        }

        self.tissue = self.attributes["GTExPortal"]["Tissue"]

        self.output_meth = os.path.join(path__methylation, self.tissue)

        path__methylation_tensor = os.path.join(self.output_meth, f'tensor/tensor_{self.tissue.lower()}.pt')
        path__methylation_probes = os.path.join(self.output_meth, f'probes/probes.pkl')
        path__methylation_subjects = os.path.join(self.output_meth, f'subjects/subjects.pkl')

        self.meth_tensor = torch.load(path__methylation_tensor).T.to(dtype=torch.float32)
        meth_subjects = pd.read_pickle(path__methylation_subjects)
        self.meth_probes = pd.read_pickle(path__methylation_probes)

        self.lt_sid_meth = meth_subjects.apply(lambda x: '-'.join(x.split('-')[:3])).to_list()
    
    def get_probes(self, lt_probes=None):
        se_sid_meth = pd.Series(self.lt_sid_meth)
        try:
            idx = se_sid_meth[se_sid_meth == self.sid].index[0]
        except:
            #print(f"[-] Methylation doesn't have patient {self.sid}")
            return None

        rep_dict = {}
        if not lt_probes:
            rep_dict["values"] = self.meth_tensor[idx,:].unsqueeze(0)
            lt_probes = list(self.meth_probes)
        else:
            idx_probes = list(self.meth_probes[self.meth_probes.isin(lt_probes)].index)
            rep_dict["values"] = self.meth_tensor[idx,idx_probes].unsqueeze(0)
        rep_dict["probes"] = lt_probes
        return rep_dict
    
class GtexGene():
    def __init__(self, sid):
        global path__gene

        self.output_gene = path__gene

        self.subject_id = "-".join(sid.split('-')[:-1])
        
        self.attributes = {
            "GTExPortal": dfpd_GtexPortal[dfpd_GtexPortal["sample_id"] == self.sid].to_dict("records")[0]
        }

        self.tissue = self.attributes["GTExPortal"]["Tissue"]

        self.path_data = os.path.join(self.output_gene, f'X_coding_{self.tissue.lower()}_log2.csv')

    @staticmethod
    @lru_cache(maxsize=None)
    def _load_dataframe(filepath):
        return pd.read_csv(filepath)
    
    def get_expressions(self, lt_genes=None):
        df = self._load_dataframe(self.path_data)

        rep_dict = {}
        try:
            if lt_genes:
                selected_columns = [col for col in df.columns if col in lt_genes]
                df_filtered = df[selected_columns]
                data = df_filtered[df_filtered["tissue_sample_id"] == self.sid].drop("tissue_sample_id", axis=1).values[0]
                rep_dict["values"] = torch.tensor(data).unsqueeze(0)
                rep_dict["genes"] = list(df_filtered.drop("tissue_sample_id", axis=1).columns)
            else:
                data = df[df["tissue_sample_id"] == self.sid].drop("tissue_sample_id", axis=1).values[0]
                rep_dict["values"] = torch.tensor(data).unsqueeze(0)
                rep_dict["genes"] = list(df.drop("tissue_sample_id", axis=1).columns)

            return rep_dict
        except:
            return None
    
class GtexPatient(GtexSlide, GtexMethylation, GtexGene):
    def __init__(self, sid):
        GtexSlide.__init__(self, sid)
        GtexMethylation.__init__(self, sid)
        GtexGene.__init__(self, sid)

        self.sid = sid
        self.subject_id = "-".join(sid.split('-')[:-1])
        
        self.attributes = {
            "GTExPortal": dfpd_GtexPortal[dfpd_GtexPortal["sample_id"] == self.sid].to_dict("records")[0]
        }

        self.tissue = self.attributes["GTExPortal"]["Tissue"]

        self.attributes.update(
            {"pathologies" : get_pathology_array(self.sid, self.tissue)}
            )

        self.attributes.update(
            {"smoker_status": get_smoker_status(self.subject_id)}
        )

        self.attributes.update(
            {"age": get_age(self.subject_id)}
        )

class GTExMultiData():
    def __init__(self, tissue, lt_modalities=None):
        global path__methylation
        global path__histology
        global path__gene

        self.path__gene = path__gene
        self.path__meth = path__methylation
        self.path__hist = path__histology

        if lt_modalities:
            self.modalities = lt_modalities
        else:
            self.modalities = ["histology", "methylation", "gene_expression"]

        self.tissue = tissue

        self.hipt_features = ["features_mean256_cls4k", "features_mean256", "features_cls4k"]

        self.path__gene_data = os.path.join(self.path__gene, f'X_coding_{self.tissue.lower()}_log2.csv')
        self.path__hist_data = os.path.join(self.path__hist, self.tissue)
        self.path__meth_data = os.path.join(self.path__meth, self.tissue)
        
        for modality in self.modalities:
            self.get_sid(modality, out=False)

        self.get_age(out=False)
        
    def get_sid(self, modality, out=True, dim="subject"):
        if modality == "histology":
            file_path = os.path.join(self.path__hist_data, 'embeddings', 'features_mean256_cls4k.pkl')
            if not os.path.exists(file_path): 
                self.hist__sample_id = list_non_empty_dir_names(self.path__hist_data)
                self.hist__sid = pd.Series(self.hist__sample_id).apply(lambda x: '-'.join(x.split('-')[:2])).to_list()
            else:
                self.hist__sample_id = list(pd.read_pickle(file_path).index)
                self.hist__sid = pd.Series(self.hist__sample_id).apply(lambda x: '-'.join(x.split('-')[:2])).to_list()
            if out:
                if dim == "subject":
                    return self.hist__sid
                elif dim == "sample":
                    return self.hist__sample_id
                
        elif modality == "methylation":
            path__methylation_subjects = os.path.join(self.path__meth_data, 'subjects/subjects.pkl')
            meth_subjects = pd.read_pickle(path__methylation_subjects)
            self.meth__sample_id = meth_subjects.apply(lambda x: '-'.join(x.split('-')[:3])).to_list()
            self.meth__sid = meth_subjects.apply(lambda x: '-'.join(x.split('-')[:2])).to_list()
            if out:
                if dim == "subject":
                    return self.meth__sid
                elif dim == "sample":
                    return self.meth__sample_id
                
        elif modality == "gene_expression":
            df_gene = pd.read_csv(self.path__gene_data)
            self.gene__sid = list(df_gene["tissue_sample_id"])
            self.gene__sample_id = pd.Series(self.gene__sid).apply(lambda x: '-'.join(x.split('-')[:3])).to_list()
            self.gene__sid = pd.Series(self.gene__sid).apply(lambda x: '-'.join(x.split('-')[:2])).to_list()
            if out:
                if dim == "subject":
                    return self.gene__sid
                elif dim == "sample":
                    return self.gene__sample_id
    
    def get_age(self, intersec=True, samples=None, out=True):
        dfpd_age = dfpd_DonorAge.drop("Unnamed: 0", axis=1).set_index("subject_id")

        if intersec:
            samples = self.intersection(fig=False, out=True, lt_modalities=self.modalities)
        self.age = dfpd_age.loc[dfpd_age.index.intersection(list(samples))]
        if out:
            return self.age
        
    def intersection(self, fig=True, out=False, lt_modalities=None, dim="subject", split=None):
        if not lt_modalities:
            lt_modalities = self.modalities

        dct_sets = {}
        for m in lt_modalities:
            d = self.get_sid(m, dim=dim)
            dct_sets[m] = set(d) 

        if split:
            split = set(split)
            dct_sets_train = {}
            dct_sets_test = {}
            for m in lt_modalities:
                dct_sets_train[m] = dct_sets[m] - split
                dct_sets_test[m] = dct_sets[m].intersection(split)

        if fig:
            import matplotlib.pyplot as plt
            from matplotlib_venn import venn2, venn3
            import venn
            if not split:
                venn_diagram = venn.venn(dct_sets, cmap="plasma", fontsize=8, figsize=(5,5))
                plt.title(f'#{dim.capitalize()} per modality for {self.tissue}')
                plt.show()
            else:
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0] = venn.venn(dct_sets_train, ax=axs[0], cmap="plasma", fontsize=8)
                axs[0].set_title('Subjects per modality for Lung (Train)')

                axs[1] = venn.venn(dct_sets_test, ax=axs[1], cmap="plasma", fontsize=8)
                axs[1].set_title('Subjects per modality for Lung (Test)')

                plt.tight_layout()
                plt.show()
        if out:
            return set.intersection(*dct_sets.values())
    
    def _drop_duplicated_sids(self, df, modality_name):
        df_cp = df.copy()
        # Drop duplicated subjects (sometimes there are multiple samples from the same subject for the same tissue):
        # Identify duplicated indices (subjects)
        duplicated_subjects = df_cp.index[df_cp.index.duplicated(keep='first')]
        if not duplicated_subjects.empty:
            print(f"[-] Duplicated subjects found on {modality_name}. Keeping the first of each: {list(duplicated_subjects)}")
            df_cp = df_cp[~df_cp.index.duplicated(keep='first')]
            return df_cp
        else:
            return df
        
    def get_meth(self, intersec=False, split=None, filter_on=None):
        path__methylation_tensor = os.path.join(self.path__meth_data, f'tensor/tensor_{self.tissue.lower()}.pt')
        path__methylation_probes = os.path.join(self.path__meth_data, 'probes/probes.pkl')
        
        self.meth_tensor = torch.load(path__methylation_tensor).T.to(dtype=torch.float32)
        self.meth_probes = pd.read_pickle(path__methylation_probes)

        se_meth_subjects = pd.Series(self.get_sid("methylation"))

        if filter_on == "age":
            se_age = self.get_age()
            set_intersec = set(list(se_age.index))
        else:
            set_intersec = self.intersection(False, True, self.modalities)

        lt_intersec = sorted(list(set_intersec))

        rep_dict = {}
        if intersec:
            filtered_series = se_meth_subjects[se_meth_subjects.isin(lt_intersec)]
            ordered_idx = [filtered_series.index[filtered_series == subject].tolist()[0] for subject in lt_intersec if subject in filtered_series.values]
            out = self.meth_tensor[ordered_idx,:]
            rep_dict["values"] = out
            rep_dict["features"] = list(self.meth_probes)
            se_subjects = pd.Series(lt_intersec).reset_index(drop=True)
        else:
            out = self.meth_tensor[:,:]
            rep_dict["values"] = out
            rep_dict["features"] = list(self.meth_probes)
            se_subjects = se_meth_subjects
        
        if split:
            idx_train, idx_test = self._split(se_subjects, split)
            rep_dict["train"] = rep_dict["values"][idx_train,:]
            rep_dict["test"] = rep_dict["values"][idx_test,:]

            se_subjects = [se_subjects.loc[idx_train].reset_index(drop=True), se_subjects.loc[idx_test].reset_index(drop=True)]
        return rep_dict, se_subjects

    def get_gene(self, intersec=False, split=None, filter_on=None):
        df_gene = pd.read_csv(self.path__gene_data).drop("tissue_sample_id", axis=1)
        df_gene = df_gene.set_index(pd.Series(self.get_sid("gene_expression")))
        rep_dict = {}

        if filter_on == "age":
            se_age = self.get_age()
            set_intersec = set(list(se_age.index))
        else:
            set_intersec = self.intersection(False, True, self.modalities)
        lt_intersec = sorted(list(set_intersec))
        if intersec:
            df_gene_filtered = df_gene.loc[lt_intersec]
            se_subjects = pd.Series(list(df_gene_filtered.index)).reset_index(drop=True)
            out = torch.tensor(df_gene_filtered.values, dtype=torch.float32)
            rep_dict["values"] = out
            rep_dict["features"] = list(df_gene.columns)
        else:
            se_subjects = pd.Series(list(df_gene.index)).reset_index(drop=True)
            out = torch.tensor(df_gene.values, dtype=torch.float32)
            rep_dict["values"] = out
            rep_dict["features"] = list(df_gene.columns)

        if split:
            idx_train, idx_test = self._split(se_subjects, split)
            rep_dict["train"] = rep_dict["values"][idx_train,:]
            rep_dict["test"] = rep_dict["values"][idx_test,:]

            se_subjects = [se_subjects.loc[idx_train].reset_index(drop=True), se_subjects.loc[idx_test].reset_index(drop=True)]
        return rep_dict, se_subjects

    def get_hist(self, intersec=False, split=None, lt_features=None, filter_on=None, force_reload=False):
        # Get HIPT features form {TISSUE_TYPE} WSIs:
        if lt_features:
            hipt_features = lt_features
        else:
            hipt_features = self.hipt_features

        rep_dict = {}
        if split:
            rep_dict["train"] = {}
            rep_dict["test"] = {}
        for feature in hipt_features:
            folder_path = os.path.join(self.path__hist_data, 'embeddings')
            file_path = os.path.join(self.path__hist_data, 'embeddings', f'{feature}.pkl')
            if not os.path.exists(file_path) or force_reload: 
                print(f"[+] Loading features {feature}...")
                lt_wsi = []
                lt_sid_hist = []
                for sid in tqdm(self.hist__sample_id):
                    try:
                        p = os.path.join(self.path__hist_data, sid, 'embeddings', 'asset_dict.pth')
                        asset_dict = torch.load(p)
                        patch_list = []
                        for patch in asset_dict.keys():
                            tensor_current = asset_dict[patch][feature]
                            patch_list.append(tensor_current)
                        lt_wsi.append(torch.cat(patch_list, dim=0).mean(dim=0).unsqueeze(0))
                        lt_sid_hist.append(sid)
                    except:
                        print(f'[-] Problems loading {sid} folder.')
                rep_dict[feature] = torch.cat(lt_wsi, dim=0)

                if not os.path.exists(folder_path):
                    Path(folder_path).mkdir(parents=True, exist_ok=True)

                df_hist = pd.DataFrame(rep_dict[feature].detach().numpy(), index=lt_sid_hist)
                df_hist.to_pickle(file_path)
                se_subjects = pd.Series(lt_sid_hist).reset_index(drop=True)
            else:
                data = pd.read_pickle(file_path)
                rep_dict[feature] = torch.tensor(data.values, dtype=torch.float32)
                se_subjects = pd.Series(list(data.index)).reset_index(drop=True)

            if filter_on == "age":
                se_age = self.get_age()
                set_intersec = set(list(se_age.index))
            else:
                set_intersec = self.intersection(False, True, self.modalities)
            lt_intersec = sorted(list(set_intersec))
            if intersec:
                se_hist_sid = pd.Series(self.hist__sid).astype(str)
                
                df_temp = pd.DataFrame(rep_dict[feature].detach().numpy(), index=se_hist_sid)

                df_temp = df_temp.loc[lt_intersec]

                df_temp = self._drop_duplicated_sids(df_temp, "histology")

                se_subjects = pd.Series(df_temp.index).reset_index(drop=True)
                rep_dict[feature] = torch.tensor(df_temp.values, dtype=torch.float32)

            if split:
                idx_train, idx_test = self._split(se_subjects, split)
                rep_dict["train"][feature] = rep_dict[feature][idx_train,:]
                rep_dict["test"][feature] = rep_dict[feature][idx_test,:]

                se_subjects = [se_subjects.loc[idx_train].reset_index(drop=True), se_subjects.loc[idx_test].reset_index(drop=True)]
        return rep_dict, se_subjects
    
    def _split(self, indices, custom_test_set):
        idx_train = indices[~indices.isin(custom_test_set)].index
        idx_test = indices[indices.isin(custom_test_set)].index
        return idx_train, idx_test
    
    def get_labels(self, se_subjects, label):
        if label=="age":
            dct_age = self.age.to_dict()["age"]
            return se_subjects.map(dct_age)

