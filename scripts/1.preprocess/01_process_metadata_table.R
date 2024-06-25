#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro@gmail.com
# @Description: Code to process the metadata
# @software version: R=4.2.2

# Load libraries
library(tidyverse)

#Set working dir 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load the dataset downloaded from GTEx
methylation_epic_array_metadata <- read_tsv(file = "../../metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv")

# Load GTEx metadat (add information about BMI, sex, age and ETHCTY)
## Note, the individuals removed from v8 were not considered in this study
gtex_metadata <- readRDS("../../metadata/GTExSampleData.RDS")

# Summarise at sample level
gtex_metadata.individual_level <- gtex_metadata %>% 
  select(SUBJID, SMTSD, SEX, AGE, ETHNCTY, BMI) %>% 
  distinct(SUBJID, .keep_all = T)

# Merge the 2 dataset
# Match by Donor ID 
merged.metadata <- merge(methylation_epic_array_metadata, gtex_metadata.individual_level, by.x = "Collaborator Participant ID", by.y = "SUBJID")

merged.metadata <- merged.metadata %>% 
  rename(SUBJID = `Collaborator Participant ID`) %>% 
  rename(Sample_ID = `Sample ID for data sharing and public release`) %>%
  rename(tissue = `Tissue Site Detail`) %>%
  select(SUBJID, Sample_ID, tissue, SEX, AGE, ETHNCTY, BMI)

#smoker_annotation_per_donor <- read.csv(file = "../../metadata/smoker_annotation_final.csv", sep = ";") %>% select(SUBJID, SmokerStatus)

#merged.metadata <- merge(merged.metadata, smoker_annotation_per_donor, all.x = T) %>% 
#  mutate(SmokerStatus = ifelse(is.na(SmokerStatus), "unknown", SmokerStatus))

#Save this metadata
write.csv(merged.metadata, file = "../../metadata/metadata.epic.noPEER.csv")

## Break the data per tissue and add histological annotation
### Lung 

merged.metadata.lung <- merged.metadata %>% 
  filter(tissue  == "Lung")

#Load histological annotation from lung and process it

lung_histo_annotation <- read.csv(file = "../../metadata/gtex_histo_annotation_raw/Gtex_portal_lung_raw_histological_annotation.csv") #<- GTEx histological metadata


lung_histo_unique_phenotype <- c()
for (disease in lung_histo_annotation$Pathology.Categories){
  for (dis in  strsplit(disease, ",")[[1]]){
    dis <- gsub(" ", "", dis)
    lung_histo_unique_phenotype <- c(lung_histo_unique_phenotype,dis)
  }
}

for (disease in lung_histo_unique_phenotype){
  lung_histo_annotation[[disease]] <- ifelse(grepl(disease, lung_histo_annotation$Pathology.Categories), 1, 0)
}

write_tsv(lung_histo_annotation, file = "../../metadata/lung_histo_annotation_parsed.tsv")

#Merge with the tissue metadata 
merged.metadata.lung$tissue_sample_id <- gsub("-SM-.+", "", merged.metadata.lung$Sample_ID)


lung_histo_annotation <- lung_histo_annotation %>%
  select(-c(Subject.ID, Sex,  Age.Bracket, Hardy.Scale, Pathology.Categories, Pathology.Notes))


lung_annotation <- merge(merged.metadata.lung, lung_histo_annotation, by.x = "tissue_sample_id", by.y = "Tissue.Sample.ID")

write.csv(lung_annotation, file = "../../metadata/lung_annotation_meth.csv")



### Ovary 

merged.metadata.ovary <- merged.metadata %>% 
  filter(tissue  == "Ovary")

#Load histological annotation from ovary and process it

ovary_histo_annotation <- read.csv(file = "../../metadata/gtex_histo_annotation_raw/Gtex_portal_ovary_raw_histological_annotation.csv")


ovary_histo_unique_phenotype <- c()
for (disease in ovary_histo_annotation$Pathology.Categories){
  for (dis in  strsplit(disease, ",")[[1]]){
    dis <- gsub(" ", "", dis)
    ovary_histo_unique_phenotype <- c(ovary_histo_unique_phenotype,dis)
  }
}

for (disease in ovary_histo_unique_phenotype){
  ovary_histo_annotation[[disease]] <- ifelse(grepl(disease, ovary_histo_annotation$Pathology.Categories), 1, 0)
}

write_tsv(ovary_histo_annotation, file = "../../metadata/ovary_histo_annotation_parsed.tsv")

#Merge with the tissue metadata 
merged.metadata.ovary$tissue_sample_id <- gsub("-SM-.+", "", merged.metadata.ovary$Sample_ID)


ovary_histo_annotation <- ovary_histo_annotation %>%
  select(-c(Subject.ID, Sex,  Age.Bracket, Hardy.Scale, Pathology.Categories, Pathology.Notes))


ovary_annotation <- merge(merged.metadata.ovary, ovary_histo_annotation, by.x = "tissue_sample_id", by.y = "Tissue.Sample.ID")

write.csv(ovary_annotation, file = "../../metadata/ovary_annotation_meth.csv")

### Merge omics metadata table 

data_multi_omics <- read.csv(file = "../../metadata/sample_ids_multiomics.csv")
data_telemores <- read.csv("../../data/telemores_data.csv")

data_telemores$sample_id  <- gsub("-SM", "", data_telemores$CollaboratorSampleID)

data_multi_omics$telemore <- ifelse(data_multi_omics$sample_id %in% data_telemores$sample_id, 1, 0)

write.csv(data_multi_omics, "../../metadata/sample_ids_multiomics_updated_tl_data.csv")
