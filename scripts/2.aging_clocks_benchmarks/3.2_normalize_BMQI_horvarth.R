#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Code to run Normalization in the data. This is an intermediary file for the analysis
# @software version: R=4.2.2

#Set path 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

#Load libraries for ploting 
library(tidyverse)
library(impute)
library(data.table)

# Normalization function
source("normalization.R")

## Load CpGs

cpgs <- read.csv(file = "AltumAge/multi_platform_cpgs.csv")
cpgs <- cpgs$X0

## Load 21k Annotation (used for normalization)
probeAnnotation21kdatMethUsed <- read.csv("data/probeAnnotation21kdatMethUsed.csv")
probeAnnotation21kdatMethUsed <- probeAnnotation21kdatMethUsed %>% filter(Name %in% cpgs)

## Tissues
tissues <- c("lung", "colon", "ovary", "prostate")

for (tissue in tissues){
  ## Create folder for intermediary files 
  
  meth <- fread(file = paste0("../../data/methylation_", tissue, ".csv"))
  
  #Load metadata 
  metadata <- read.csv(file = "../../metadata/eGTExDNA_Pierce_Jan18.09-11-2021.tsv", sep = "\t")
  if (tissue == "colon"){
    tissue_2 <- "Colon - Transverse"
  }else{
    tissue_2 <- str_to_title(tissue)
  }
  metadata <- metadata %>% filter(`Tissue.Site.Detail` == tissue_2)
  
  # Load age data
  age_data <- read.csv(file =  paste0("../../metadata/", tissue, "_annotation_meth.csv"))
  
  # Load test data
  meth_test_samples <- read.csv(file = paste0("../../metadata/", tissue, "_test_metadata.csv"))
  
  #Filter for samples with age data
  samples_with_age <- age_data %>% pull(Sample_ID)
  meth <- meth %>% 
    select(probe, all_of(samples_with_age))
  
  ## Missing probes
  meth <- meth %>% 
    filter(probe %in% cpgs)
  
  match1 <- match(cpgs , meth$probe)
  sum(is.na(match1))
  
  missing_probes <- cpgs[is.na(match1)]
  missing_probes_data <- matrix(data = NA, nrow = length(missing_probes), ncol = ncol(meth))
  missing_probes_data[,1] <- missing_probes
  colnames(missing_probes_data) <-  colnames(meth)
  
  missing_probes_data <- as.data.frame(missing_probes_data)
  
  meth <- rbind(meth, missing_probes_data)
  meth <- meth %>% 
    as_tibble() %>%
    mutate_at(2:ncol(meth), as.numeric)
  
  ### Data imputation
  meanMethBySample = as.numeric(apply(as.matrix(meth[,-1]),2,mean,na.rm=TRUE))
  minMethBySample = as.numeric(apply(as.matrix(meth[,-1]),2,min,na.rm=TRUE))
  maxMethBySample = as.numeric(apply(as.matrix(meth[,-1]),2,max,na.rm=TRUE))
  
  datMethUsed= t(meth[,-1])
  colnames(datMethUsed)=as.character(meth$probe)
  
  noMissingPerSample=apply(as.matrix(is.na(datMethUsed)),1,sum)
  table(noMissingPerSample)
  
  if (max(noMissingPerSample,na.rm=TRUE)<3000 ){
    if ( max(noMissingPerSample,na.rm=TRUE)>0 ){
      dimnames1=dimnames(datMethUsed)
      datMethUsed= data.frame(t(impute.knn(t(datMethUsed))$data))
      dimnames(datMethUsed)=dimnames1
    } # end of if
  } # end of if (! fastImputation )
  
  datMethUsedNormalized <- BMIQcalibration(
    datM = datMethUsed,
    goldstandard.beta = probeAnnotation21kdatMethUsed$goldstandard2,
    plots = FALSE
  )
  
  write.csv(datMethUsedNormalized, file = paste0("../../results/2.epigenetic_clocks/AltumAge/", tissue, "_BQMI_normalized.csv"))
}


