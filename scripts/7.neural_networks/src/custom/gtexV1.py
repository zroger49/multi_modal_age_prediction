import os
from histolab.slide import Slide
from pathlib import Path
import pandas as pd
import numpy as np
from ..gen_patch.create_patches_fp import patch_one_wsi
import openslide
import yaml
import torch

#os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
#print("PYTORCH_ENABLE_MPS_FALLBACK: ", os.environ["PYTORCH_ENABLE_MPS_FALLBACK"])

# =====================
# INITIALIZE HIPT
# =====================
from src.HIPT_4K.hipt_4k import HIPT_4K
from src.HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from src.HIPT_4K.hipt_heatmap_utils import *


light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)

pretrained_weights256 = './src/HIPT_4K/Checkpoints/vit256_small_dino.pth'
pretrained_weights4k = './src/HIPT_4K/Checkpoints/vit4k_xs_dino.pth'
device256 = torch.device('mps')
device4k = torch.device('mps')

### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

### ViT_256 + ViT_4K loaded into HIPT_4K API
model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
model.eval()
# =====================

import yaml
def get_parameters():
    with open("./conf/params__global.yml") as params:
        params_dict = yaml.safe_load(params)
    return params_dict

params = get_parameters()

path__gtex_portal_data = params["PATH__GTExPortalDB"]
path__smoker_annotation = params["PATH__SmokerStatus"]
path__donor_age = params["PATH__DonorAge"]
#path__test_sample = params["PATH_TestSample"].format(params["TISSUE_TYPE"])

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
    def __init__(self, sid, output):
        self.sid = sid
        self.sample_id = "-".join(sid.split('-')[:-1])
        self.output = output
        self.tile_path = f'{self.output}/{self.sid}'
        self.slide_path = f'{self.output}/{self.sid}.svs'
        
        self.attributes = {
            "GTExPortal": dfpd_GtexPortal[dfpd_GtexPortal["sample_id"] == self.sid].to_dict("records")[0]
        }

        self.tissue = self.attributes["GTExPortal"]["Tissue"]

        self.attributes.update(
            {"pathologies" : get_pathology_array(self.sid, self.tissue)}
            )

        self.attributes.update(
            {"smoker_status": get_smoker_status(self.sample_id)}
        )

        self.attributes.update(
            {"age": get_age(self.sample_id)}
        )

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

    def get_tiles_path(self, prefix_path=None):
        if prefix_path:
            self.prefix_path = f"{self.tile_path}/{prefix_path}"
        
        path = f"{self.prefix_path}"
        return os.listdir(path)