#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Code to split the DNA methylation data per tissue
# @software version: R=4.2.2

# Code based on https://github.com/Mele-Lab/2023_GTEx_Smoking/blob/main/analysis/scripts/12.Split_data.R

library(data.table)
library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


dir.create("../../results/3.feature_selection/data/")


data_path <- "../../data/GSE213478_methylation_DNAm_noob_final_BMIQ_all_tissues_987.txt.gz" #Path where I have downloaded the raw beta values from https://www.nature.com/articles/s41588-022-01248-z 
print("Reading whole data")
data <-  fread(data_path)

#split data into tissues based on metadata
metadata <- read.delim("../../metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv") #Data downloaded from Oliva et al.
names(metadata)[1] <- "ID"
names(metadata)[15] <- "Tissue"

samples <- colnames(data)[2:ncol(data)]

tissues <- c("Lung", "Ovary")

## Save data 
for(tissue in tissues){
  print(tissue)
 
 
 
  tissue_name <- tissue_name_2 <- tissue_name_3 <- tissue
  
  #dir.create(paste0("../../results/3.feature_selection//data/", tissue_name), recursive = T)
  
  samples <- metadata %>% 
    filter(Tissue == tissue) %>% 
    pull(ID)
  
  metadata.tissue <- metadata %>% 
    filter(Tissue == tissue) %>% 
    select(ID, Collaborator.Participant.ID)
  
  data_to_save <- data %>% 
    select(V1, all_of(samples))
  

  #Add data
  donor_metadata <- read.delim("../../metadata/metadata_to_find_biomarkers/GTEx_Subject_Phenotypes.GRU.txt.gz")
  donor_metadata <- donor_metadata[,c("SUBJID", "SEX", "AGE", "BMI", "TRISCHD", "DTHHRDY")]
  metadata_to_save <- merge(donor_metadata, metadata.tissue,by.y = "Collaborator.Participant.ID", by.x = "SUBJID")
  #metadata_to_save <- merge(metadata_to_save, donor_metadata, by="SUBJID")
  
  saveRDS(metadata_to_save, file = paste0("../../results/3.feature_selection/data/", tissue_name, "/methylation_metadata.rds"))
  

  #Data to save only for all donors in the given tissue:
  colnames(data_to_save)[2:ncol(data_to_save)] <- sapply(colnames(data_to_save)[2:ncol(data_to_save)], function(x) paste0(strsplit(x, "-")[[1]][1:2], collapse = "-")) #Now the column name is the subject id
  data_to_save <- data_to_save %>% 
    select(V1, all_of(metadata_to_save$SUBJID))
  saveRDS(data_to_save, file = paste0("../../results/3.feature_selection/data/", tissue_name, "/methylation_data.rds"))
}
