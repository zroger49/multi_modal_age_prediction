#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Code to run differential gene expression analysis
# @software version: R=4.2.2


setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(tidyverse)
suppressMessages(library(edgeR)) #Already includes limma

tissues <- c("Lung", "Ovary")
sex_tissues <- c("Ovary")


for (tissue in tissues){
  
  tissue_name = tissue_name_2 = "lung"
  
  if (!dir.exists(paste0("../../results/3.feature_selection_gene_expresion//results/", tissue))){
    dir.create(paste0("../../results/3.feature_selection_gene_expresion/results/", tissue), recursive = T)
  }
  
  all_folds = c(0:4, "train")
  
  cat("Analysing ", tissue, "\n")
  
  # Load counts
  counts <- readRDS(paste0("../../results/3.feature_selection_gene_expresion//data/", tissue_name, "/gexp_data.rds")) %>% as.data.frame()
  
  # Load TPM 
  logTPM <- read.csv(paste0("../../data/X_coding_", tissue_name, "_log2.csv")) 
  row.names(logTPM) <- logTPM$tissue_sample_id
  logTPM$tissue_sample_id <- NULL
  logTPM <- logTPM %>% t() %>% as.data.frame()
  
  for (fold in all_folds){
    cat(paste0("Fold ", fold, "\n"))
    cat("Reading data ...")
    
    if (fold != "train"){
      dir_res <- paste0("/fold_", fold)
      if (!dir.exists(paste0("../../results/3.feature_selection_gene_expresion//results/", tissue, dir_res))){
        dir.create(paste0("../../results/3.feature_selection_gene_expresion//results/", tissue, "/fold_", fold), recursive = T)
      }
    }else{
      dir_res <- paste0("/train")
      if (!dir.exists(paste0("../../results/3.feature_selection_gene_expresion/results/", tissue, dir_res))){
        dir.create(paste0("../../results/3.feature_selection_gene_expresion/results/", tissue, dir_res), recursive = T)
      }
    }
    
    # Reading metadata
    cat("Reading metadata...")
    metadata <- readRDS(paste0("../../results/3.feature_selection_gene_expresion/data/", tissue_name, "/gene_expression_metadata.rds"))
    metadata$ID <- gsub("-SM-.*{4,5}", "", metadata$SAMPID)
    
    covariates <- c("SEX", "AGE", "BMI", "SMRIN", "SMTSISCH", "DTHHRDY")
    
    metadata$SEX <- as.factor(metadata$SEX)
    metadata$DTHHRDY <- as.factor(metadata$DTHHRDY)
    
    rownames(metadata) <- metadata$SUBJID
    metadata$SUBJID <- NULL
    
    #If sexual tissue, we remove the covariate sex    
    if(tissue %in% sex_tissues){ 
      covariates <- covariates[covariates!="SEX"]
    }
    
    
    if (fold != "train"){
      cat("Selecting fold Sample \n")
      fold_data <- read.csv(paste0("../../results/3.feature_selection_gene_expresion//", tissue_name_2, "/fold_", fold, "_train.csv"))
      metadata <- metadata %>% filter(ID %in% fold_data$sample)
      metadata$sample_id <- gsub("-SM-.*", "", metadata$ID)
      
    }else{
      #Remove the test set samples 
      cat("Removing the test samples\n")
      test_set <- read.csv(file = paste0("../../metadata/", tissue_name, "_test_metadata.csv"))
      metadata$sample_id <- gsub("-SM-.*", "", metadata$ID)
      metadata <- metadata %>% 
        filter(!sample_id %in% test_set$sample_id)
      
      metadata <- metadata %>% 
        filter(sample_id %in% colnames(logTPM))
    }
    
    metadata_2 <- metadata[, covariates]
    
    ## Keep only samples from the train set with age data
    data_train <- counts %>% select(Name, all_of(row.names(metadata)))
    logTPM_train <- logTPM %>% select(metadata$sample_id)
    
    gene_names <- data_train$Name
    data_train <- data_train %>% select(-Name)
    row.names(data_train) <- gene_names
    
    print(dim(data_train))
    print(dim(logTPM_train))
    print(dim(metadata_2))
    
    #  Normalize gene expression distributions:
    # Create DGEList object
    dge <- DGEList(data_train)
    
    # Calculate normalization factors (does not do the normalization yet, only computes the factors)
    dge <- calcNormFactors(dge)
    
    # Voom
    v <- voom(dge, design = NULL, normalize = "quantile", save.plot=F, plot = F) 
    
    # Limma function ####
    limma_lm <- function(fit, covariate, covariate_data){ #It returns the differentially expressed genes for a particular variable based on a limma fit object
      covariate <<- covariate  #makeContrast does not read the function's environment, so I add covariate to the general environment in my session
      contrast.matrix <- suppressWarnings(makeContrasts(covariate, levels=fit$design)) #Warnings due to change of name from (Intercept) to Intercept
      fitConstrasts <- suppressWarnings(contrasts.fit(fit,contrast.matrix))
      eb = eBayes(fitConstrasts)
      tt.smart.sv <- topTable(eb,adjust.method = "BH",number=Inf)
      return(tt.smart.sv)
    }
    
    
    model_function <- function(mod, mod_no_age){
      cat("...Modelling... ")
      fit_M <- lmFit(v, mod_no_age)
      residuals <- resid(fit_M, logTPM_train)
      saveRDS(residuals, paste0("../../results/3.feature_selection_gene_expresion//results/", tissue, dir_res, "/gexp_residuals.rds")) #Correcting for covariates without Age
      
      
      #Cor with age (residuals)
      cor_with_age <- cor(t(residuals), metadata_2$AGE, method = "spearman")
      colnames(cor_with_age) <- "rho"
      write.csv(cor_with_age, file =  paste0("../../results/3.feature_selection_gene_expresion/results/", tissue, dir_res, "/cor_age_residuals.csv"), quote = F, row.names = T)
      
      #Cor with Age (logTPM values)
      cor_with_age <- cor(t(logTPM_train), metadata_2$AGE, method = "spearman")
      colnames(cor_with_age) <- "rho"
      write.csv(cor_with_age, file =  paste0("../../results/3.feature_selection_gene_expresion/results/", tissue, dir_res, "/cor_age_values.csv"), quote = F, row.names = T)
      
      
      #summary(decideTests(fit_M))
      
      fit_M <- lmFit(v, mod)
      res_M <- limma_lm(fit_M, "AGE")
      
      write.csv(res_M, paste0("../../results/3.feature_selection_gene_expresion/results/", tissue, dir_res, "/DEG_results.csv"), quote = F, row.names = T)
      cat("Number of genes DEGs", sum(res_M$adj.P.Val < 0.05))
    }
    
    #Creating model
    mod <- model.matrix( as.formula(paste(" ~  ", paste(covariates,collapse = "+"))), data =  metadata)
    mod_no_age <- model.matrix( as.formula(paste0("~", paste0(covariates[covariates!="AGE"], collapse="+"))), data =  metadata)
    
    # Limma fit
    model_function(mod, mod_no_age)
    cat("\n")
  }
}


