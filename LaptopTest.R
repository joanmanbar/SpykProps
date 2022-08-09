


setwd(dirname(rstudioapi::getSourceEditorContext()$path))


# Load libraries
library(reshape2)
library(ggplot2)
library(VCA)
library(inti)
library(kableExtra)
library(tidyverse)
library(dplyr)
library(viridis)
library(hrbrthemes)
library(car)
library(agricolae)
library(emmeans)
library(multcomp)
library(lme4)
library(ggpubr)
library(GGally)
library(ggridges)
library(ggfortify)
library(factoextra)
library(rpca)
library(readxl)
library(data.table)
library(writexl)
library(STB)









df1 <- read.csv('Output//LaptopTest_SpikeData_20220624.csv')
df1$Original_Name <- df1$Image_Name
df1 <- df1[,-1]

df2 <- read.csv('LaptopTest.csv')

df1 <- merge(df2, df1)
df1$CodeComb <- paste0(df1$Image_Name,df1$Spike_Label)

spk <- df1[df1$Spike_Label==1, ]
spk <- df1[df1$Scanner=="SCN1", ]


ggplot(spk, aes(x = Cable, y = Area, fill = Laptop)) +
  geom_boxplot() + geom_jitter()

ggplot(spk, aes(x = Scanner, y = Area, fill = Laptop)) +
  geom_violin() + geom_jitter() +
  facet_wrap(~Cable) 




color_df <- spk

# Look for zero-variance columns
var0 <- unlist(lapply(color_df, function(x) 0 == var(if (is.factor(x)) as.integer(x) else x)))
var0 <- names(var0)[var0 == TRUE]
# df = color_df[ , -which(names(color_df) %in% var0)]
var0 <- c("Blue_min","H_min","S_max","S_min")

color_df = color_df[ , -which(names(color_df) %in% var0)]

df <- color_df
M <- df[,14:113]
M <- df[, c('Red_mean','Green_mean','Blue_mean')]

FO_pca2 <- prcomp(M, scale=TRUE)


pca_plot <- autoplot(FO_pca2, x=1,y=2,data = df, colour = 'Cable',
                     loadings = F, loadings.colour = 'white',
                     loadings.label = F, loadings.label.size = 3)
pca_plot




