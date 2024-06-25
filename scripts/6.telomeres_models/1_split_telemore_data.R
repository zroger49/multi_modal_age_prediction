#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Divide Telemore samples per tissue
# @software version: R=4.2.2

# Set working dir
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


# Load data
## Tl data
tl_data <- read.csv(file = "../../data/telemores_data.csv") # Data can be downloaded from  (https://www.gtexportal.org/home/downloads/egtex/telomeres)

# GTEx data
gtex <-  readRDS("../../metadata/GTExSampleData.RDS") %>% select(SUBJID, AGE, BMI, SEX, RACE) %>% distinct(.keep_all = T)

tissues <- c("Lung", "Ovary")
tissue_name <- c("lung", "ovary")


for (i in 1:length(tissues)){
  name <- tissue_name[i]
  file_name = paste0("../../data/telemores_", name, ".csv")
  data <- tl_data %>% filter(TissueSiteDetail == tissues[i])
  
  print(tissues[i])
  print(dim(data))
  
  data <- merge(data, gtex, by.x = "CollaboratorParticipantID", by.y = "SUBJID")
  
  print(dim(data))
  write.csv(data, file_name)
}

