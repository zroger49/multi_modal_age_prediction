#!/usr/bin/env Rscript
# @Author: Rogério Eduardo Ramos Ribeiro
# @E-mail: rogerio.e.ramos.ribeiro
# @Description: Code to analyse the missing probes data
# @software version: R=4.2.2

library(tidytree)
library(data.table)

# Set working dir 
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load lung data as an example
lung <- fread(file = "../../data/methylation_lung.csv")

## Load Horvarh data
probeAnnotation21kdatMethUsed=read.csv("data/probeAnnotation21kdatMethUsed.csv")
probeAnnotation27k=read.csv("data/datMiniAnnotation27k.csv")
datClock=read.csv("data/AdditionalFile3.csv")

datClock_missing <- datClock %>% 
  filter(!CpGmarker %in% lung$probe)



p1 <- hist(datClock$CoefficientTraining, breaks = 20)
p2 <- hist(datClock_missing$CoefficientTraining, breaks = 20)

png("../../aging_notes/figures/2.epigenetic_clocks/missing_data_horvarth.png")
plot( p1, col=rgb(0,0,1,1/4)) 
plot( p2, col=rgb(1,0,0,1/4), add=T)
dev.off()


## Load Hannum data
datClock=read.csv("data/hanmum_probes.txt", sep = "\t", dec = ",")

datClock_missing <- datClock %>% 
  filter(!Marker %in% lung$probe)


p1 <- hist(datClock$Coefficient, breaks = 20)
p2 <- hist(datClock_missing$Coefficient, breaks = 20)

png("../../aging_notes/figures/2.epigenetic_clocks/missing_data_hannum.png")
plot( p1, col=rgb(0,0,1,1/4)) 
plot( p2, col=rgb(1,0,0,1/4), add=T)
dev.off()

## AltumAge
#...