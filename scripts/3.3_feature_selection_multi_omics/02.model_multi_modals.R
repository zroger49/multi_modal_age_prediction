#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Code to run differential methylation analysis
# @software version: R=4.2.2

# Code based on https://github.com/Mele-Lab/2023_GTEx_Smoking/blob/main/analysis/scripts/12.to_submit_model.sh
library(tidyverse)
library(limma)
library(caret)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

tissues <- c("Lung", "Ovary")

# Load metadata 
metadata.all <- read.delim("../../metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv") #Data downloaded from Oliva et al.
names(metadata.all)[1] <- "ID"
names(metadata.all)[15] <- "Tissue"


for (tissue in tissues){
  
  if (tissue == "Lung"){
    tissue_name = tissue_name_2 = "lung"
  }else if(tissue == "Ovary"){
    tissue_name = tissue_name_2 = "ovary"
  }
  
  # Load all the samples
  metadata.tissue <- metadata.all %>% 
    filter(Tissue == tissue) %>% 
    select(ID, Collaborator.Participant.ID)
  
  if (!dir.exists(paste0("../../results/3.feature_selection_multimodal//results/", tissue))){
    dir.create(paste0("../../results/3.feature_selection_multimodal//results/", tissue), recursive = T)
  }
  
  cat("Analysing ", tissue, "\n")
  for (fold in c(0,1,2,3,4, "train")){
    cat(paste0("Fold ", fold, "\n"))
    cat("Reading data ...")
    
    if (fold != "train"){
      dir_res <- paste0("/fold_", fold)
      if (!dir.exists(paste0("../../results/3.feature_selection_multimodal/results/", tissue, dir_res))){
        dir.create(paste0("../../results/3.feature_selection_multimodal/results/", tissue, "/fold_", fold), recursive = T)
      }
    }else{
      dir_res <- paste0("/train")
      if (!dir.exists(paste0("../../results/3.feature_selection_multimodal/results/", tissue, dir_res))){
        dir.create(paste0("../../results/3.feature_selection_multimodal/results/", tissue, dir_res), recursive = T)
      }
    }
    
    data <- readRDS(paste0("../../results/3.feature_selection/data/", tissue, "/methylation_data.rds")) #From whole compressed data in 5.6G to compressed 1.4G/1.1Gb only in Lung (the highest number of samples)
      
    cat("Reading metadata...")
    metadata <- readRDS(paste0("../../results/3.feature_selection/data/", tissue, "/methylation_metadata.rds"))
    
    if (fold != "train"){
      cat("Selecting fold Sample \n")
      fold_data <- read.csv(paste0("../../results/3.feature_selection_multimodal/", tissue_name_2, "/fold_", fold, "_train.csv"))
      exclusive_samples <- read.csv(paste0("../../results/3.feature_selection_multimodal/", tissue_name_2, "/exclusive_methylation_train.csv"))
      samples <- c(fold_data$sample, exclusive_samples$Sample_ID)
      metadata <- metadata %>% filter(ID %in% samples)
      metadata$sample_id <- gsub("-SM-.*", "", metadata$ID)
      
    }else{
      #Remove the test set samples : Note I use the complete train set here
      cat("Removing the test samples\n") 
      test_set <- read.csv(file = paste0("../../metadata/", tissue_name, "_test_metadata.csv"))
      metadata$sample_id <- gsub("-SM-.*", "", metadata$ID)
      metadata <- metadata %>% 
        filter(!sample_id %in% test_set$sample_id)
    }
    
    ## Also filter samples that were removed from GTEx 
    # - Merge with the clinical traits sample
    metadata.good <- read.csv(file = paste0("../../metadata/", tissue_name, "_annotation_meth.csv"))
    
    metadata <- metadata %>% 
      filter(sample_id %in% metadata.good$tissue_sample_id)
    
    
    ## Keep only samples from the train set with age data
    data_train <- data %>% select(all_of(metadata$SUBJID)) 
    
    metadata$SEX <- as.factor(metadata$SEX)
    metadata$DTHHRDY <- as.factor(metadata$DTHHRDY)
    
    if(length(levels(metadata$SEX))==1){
      metadata <- metadata[,-which(names(metadata) == "SEX")]
      individual_variables <- c("AGE", "BMI", "TRISCHD", "DTHHRDY")
    } else{
      individual_variables <- c("AGE", "BMI", "TRISCHD", "DTHHRDY", "SEX")
    }
    rownames(metadata) <- metadata$SUBJID
    metadata$SUBJID <- NULL
    
    probes <- data$V1
    data$V1 <- NULL
    
    metadata_2 <- metadata[, individual_variables]
    names(metadata_2)
    
    #Use limma to create my contrasts, as other build in functions doesn't allow me to modify it
    limma_function <- function(fit, x){
      covariate <<- x #makeContrast does not read the function's environment, so I add covariate to the general environment in my session
      contrast.matrix <- suppressWarnings(makeContrasts(covariate, levels=fit$design)) #Warnings due to change of name from (Intercept) to Intercept
      fitConstrasts <- suppressWarnings(contrasts.fit(fit, contrast.matrix)) #Warning due to Intercept name
      eb = eBayes(fitConstrasts)
      tt.smart.sv <- topTable(eb,adjust.method = "BH",number=Inf)
      return(tt.smart.sv)
    }
    
    data_train <- sapply(data_train, as.numeric)
    rownames(data_train) <- probes
    M <- log2(data_train/(1-data_train)) # M is better for differential although beta should be plotted for interpretation
    
    print(dim(M))
    
    metadata_no_age <- metadata_2 %>% 
      select(-AGE)
    
    mod <- model.matrix( as.formula(paste0("~", paste0(colnames(metadata_2), collapse="+"))), data =  metadata_2)
    mod_no_age <- model.matrix( as.formula(paste0("~", paste0(colnames(metadata_no_age), collapse="+"))), data =  metadata_no_age)
    
    
    model_function <- function(mod, mod_no_age){
      cat("...Modelling... ")
      #fit_M <- lmFit(M, mod_no_age)
      #residuals <- resid(fit_M, M)
      #saveRDS(residuals, paste0("../../results/3.feature_selection_multimodal/results/", tissue, dir_res, "/methylation_residuals.rds")) #Correcting for covariates without Age
      #Cor with age (residuals)
      #cor_with_age <- cor(t(residuals), metadata_2$AGE, method = "spearman")
      #colnames(cor_with_age) <- "rho"
      #write.csv(cor_with_age, file =  paste0("../../results/3.feature_selection_multimodal/results/", tissue, dir_res, "/cor_age_residuals.csv"), quote = F, row.names = T)
      
      #Cor with Age (Methylation values)
      #cor_with_age <- cor(t(M), metadata_2$AGE, method = "spearman")
      #colnames(cor_with_age) <- "rho"
      #write.csv(cor_with_age, file =  paste0("../../results/3.feature_selection_multimodal/results/", tissue, dir_res, "/cor_age_M_values.csv"), quote = F, row.names = T)
      
      #summary(decideTests(fit_M))
      
      fit_M <- lmFit(M, mod)
      res_M <- limma_function(fit_M, "AGE")
      
      write.csv(res_M, paste0("../../results/3.feature_selection_multimodal//results/", tissue, dir_res, "/DML_results.csv"), quote = F, row.names = T)
    }
    
    model_function(mod, mod_no_age)
    cat("\n")
  }
}


