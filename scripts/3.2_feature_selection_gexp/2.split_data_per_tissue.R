#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Code to split the Gene expression data per tissue
# @software version: R=4.2.2

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Libraries
library(data.table)
library(tidyverse)
library(RColorBrewer)
library(ComplexHeatmap)


#split data into tissues based on metadata
metadata <- readRDS("../../metadata/GTExSampleData.RDS")

tissues <- c("Lung", "Ovary")

covariates_individual <- c("COHORT", "AGE", "ETHNCTY", "BMI", "DTHHRDY", "SEX")
covariates_sample <- c("SMRIN", "SMTSISCH")

corr_table <- c()

for(tissue in tissues){
  print(tissue)
  
  
  tissue_name <- tissue
  tissue_name_2 <- tolower(tissue)
  
  if (!dir.exists(paste0("../../results/3.feature_selection_gene_expresion/data/", tissue_name_2))){
    dir.create(paste0("../../results/3.feature_selection_gene_expresion/data/", tissue_name_2), recursive = T)
  }  
  
  samples <- metadata %>% 
    filter(SMTSD == tissue) %>% 
    pull(SAMPID)
  
  print(length(samples))
  
  # Remove test dataset
  ## If we filter based on this set it can be considered data leakage
  test_set <- read.csv(paste0("../../metadata/", tissue_name_2,  "_test_metadata.csv"))
  samples_2 <- gsub("-SM-.*", "", samples)
  samples <- samples[!samples_2 %in% test_set$sample_id]
  
  metadata.tissue <- metadata %>% 
    filter(SAMPID %in% samples) %>% 
    select(SAMPID, SUBJID, all_of(covariates_sample))
  
  #Add data
  donor_metadata <- read.delim("../../metadata/GTEx_Subject_Phenotypes.GRU.txt.gz")
  donor_metadata <- donor_metadata[,c("SUBJID", covariates_individual)]
  metadata_to_save <- merge(donor_metadata, metadata.tissue ,by = "SUBJID")

  #Omit NA
  metadata_to_save <- na.omit(metadata_to_save)
  
  saveRDS(metadata_to_save, file = paste0("../../results/3.feature_selection_gene_expresion/data/", tissue_name_2, "/gene_expression_metadata.rds"))
    
  #Keep the same genes used in the gene expression models:
  log2expression_matrix <- fread(paste0("../../data/X_coding_", tissue_name_2, "_log2.csv"))
  genes <- colnames(log2expression_matrix)[-1]

  data_to_save <- data %>%
    select(Name, all_of(samples)) %>%
    filter(Name %in% genes)

  #Data to save only for all donors in the given tissue:
  colnames(data_to_save)[2:ncol(data_to_save)] <- sapply(colnames(data_to_save)[2:ncol(data_to_save)], function(x) paste0(strsplit(x, "-")[[1]][1:2], collapse = "-")) #Now the column name is the subject id

  #Remove samples to be in line with the metadata -> I do this step after gene expression filtering
  data_to_save <- data_to_save %>%
    select(Name, all_of(metadata_to_save$SUBJID))

  saveRDS(data_to_save, file = paste0("../../results/3.feature_selection_gene_expresion/data/", tissue_name_2, "/gexp_data.rds"))
}
