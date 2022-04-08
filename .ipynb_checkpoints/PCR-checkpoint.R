

# Set working directory
mydir <- dirname(rstudioapi::getSourceEditorContext()$path)
setwd(mydir)
# library(here)
# set_here(path = ".", verbose = TRUE)

# Define directories
mydir <- getwd()
IN <- paste0(mydir, '/IN')


data1 <- read.csv(paste0(IN, '/Compiled_2.csv'))


# Organize: Info, Shat, Spks, Brchs, PicInfo
# Split to get means (Optional)
# Info <- data1[, c(1:2, 39:41)]
df1 <- data1[, c(1:2, 7:11, 21:24, 29:30, 31:34,37,39:40)]
# rownames(df1) <- df1[, 'PicSpike']

# Avoid zeros so we can log transform
df1$sht_spk_NOgs <- df1$sht_spk_NOgs + 1
df1$sht_pctg_NOgs <- df1$sht_pctg_NOgs + 1

# test <- as.data.frame(df1[, 3])
# test$Scaled <- scale(test[,1])
# cor(test$`df1[, 3]`, test$Scaled)

library(ggplot2)
library(ggridges)
library(ggfortify)
library(factoextra)
library(gridExtra)
library(viridis)

# ggplot(df1, aes(x = log(sht_spk_NOgs), y = PI)) +
#   geom_density_ridges_gradient(aes(fill = ..x..), scale = 3, size = 0.3) + scale_fill_gradientn(colours = c("gold", "goldenrod2", "red"),
#     name = "Shattering") +
#   labs(y = "PI Accession", x = "Log of shattered seed/spike (g)") + 
#   theme(text = element_text(size=20), axis.text.x=element_blank())

ggplot(df1, aes(x = sht_pctg_NOgs, y = PI)) +
  geom_density_ridges_gradient(aes(fill = ..x..), scale = 3, size = 0.3, bandwidth=4) +
  scale_fill_gradientn(colours = c("gold", "goldenrod2", "red"), name = "Percentage") +
  labs(y = "PI Accession", x = "Percentage shattered from spike") +
  theme(panel.background = element_blank(), 
        text = element_text(size=20), panel.spacing = unit(0.1, "lines"))


# bp1 <- ggplot(data = df1, mapping = aes(x = PI, y = log(sht_spk_NOgs))) +
#   geom_boxplot(alpha = 0) +
#   geom_jitter(alpha = 0.3, color = "tomato") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 0.9, vjust = 0.3)) +
#   labs(y= "Log of shattered seed/spike (g)", x = "PI")
# 
# bp2 <- ggplot(data = df1, mapping = aes(x = PI, y = sht_pctg_NOgs)) +
#   geom_boxplot(alpha = 0) +
#   geom_jitter(alpha = 0.3, color = "tomato") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 0.9, vjust = 0.3)) +
#   labs(y= "Shattering proportion from whole spike (%)", x = "PI")
# 
# bp2






# Boxplots for sie and branches
bp1 <- ggplot(df1, aes(x=PI, y=scale(Spk_length), fill = PI)) + 
  geom_boxplot() + 
  theme(text = element_text(size=20), axis.text.x=element_blank()) +
  labs(y= "Spike length (SD)", x = "PI Accesson")
  

bp2 <- ggplot(df1, aes(x=PI, y=scale(Area), fill = PI)) +
  geom_boxplot() + 
  theme(text = element_text(size=20), axis.text.x=element_blank()) + 
  labs(y='Spike area (SD)', x = "PI Accesson")

grid.arrange(bp1, bp2, nrow=2)


bp1 <- ggplot(df1, aes(x=PI, y=Num_Branches, fill=PI)) +
  geom_boxplot() + 
  theme(text = element_text(size=20), axis.text.x=element_blank()) +
  labs(y= "Number of Branches", x = "PI Accesson")

bp2 <- ggplot(df1, aes(x=PI, y=scale(JE_med), fill=PI)) +
  geom_boxplot() + 
  theme(text = element_text(size=20), axis.text.x=element_blank()) +
  labs(y="Length of Primary Branches (SD)", x = "PI Accesson")

grid.arrange(bp1, bp2, nrow=2)



# library(ggpubr)
# library(magrittr)
# 
# ggscatterhist(
#   df1, x = "Spk_length", y = "Spk_width",
#   color = "PI", size = 3, alpha = 0.6,
#   margin.plot = "boxplot",
#   ggtheme = theme_bw()
# )
# 
# 



## Size and branching correlations
# cor(df1$Area,df1$






library(PerformanceAnalytics)
chart.Correlation(df1[,c(11:18)])

# Subset matrix for PCA
mat1 <- df1[,c(8:13,16:18)]
mat1 <- as.matrix(mat1)

# PCA
pca1 <- prcomp(mat1, scale = TRUE)

# Visuaize PCA

# library(openxlsx)
p <- fviz_eig(pca1, addlabels=TRUE, hjust = -0.3,
              barfill="gray", barcolor ="darkblue",
              linecolor ="red") + ylim(0, 85) + 
  theme_minimal()


pca_plot <- autoplot(pca1, data = df1, colour = 'PI',
                     loadings = TRUE, loadings.colour = 'blue',
                     loadings.label = TRUE, loadings.label.size = 3)

grid.arrange(p, pca_plot, nrow=2)













# Principal Components Regression
df2 <- as.data.frame(pca1$x)
df2$PicSpike <- df1$PicSpike
df3 <- merge(df1[, c(1:6, 20)], df2)

# Correlations
chart.Correlation(df3[,c(5:16)])



LM1 <- lm(sht_spk_NOgs ~ PI+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9, data = df3)
LM2 <- lm(sht_spk_gs ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9, data = df3)
LM3 <- lm(sht_pctg_NOgs ~ PI+PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9, data = df3) # best
# LM4 <- lm(sht_pctg_gs ~ PC1+PC2+PC3+PC4+PC5+PC6+PC7+PC8+PC9, data = df3)

summary(LM3)







library(pls)

df4 <- df1[,c(6,8:18,2)]
pcr_m1 <- pcr(sht_pctg_NOgs~., data = df4, scale = TRUE, validation = "CV")

summary(pcr_m1)

# Plot the root mean squared error
validationplot(pcr_m1)



predplot(pcr_m1)
















# Now let's use MEANS
library(dplyr)
means1 <- df1 %>% group_by(Code,PI) %>% summarize( sht_spk_NOgs = mean(sht_spk_NOgs, na.rm = TRUE),
                                sht_spk_gs = mean(sht_spk_gs, na.rm = TRUE),
                                sht_pctg_NOgs = mean(sht_pctg_NOgs, na.rm = TRUE),
                                sht_pctg_gs = mean(sht_pctg_gs, na.rm = TRUE),
                                L = mean(L, na.rm = TRUE),
                                a = mean(a, na.rm = TRUE),
                                b = mean(b, na.rm = TRUE),
                                Area = mean(Area, na.rm = TRUE),
                                Spk_length = mean(Spk_length, na.rm = TRUE),
                                Spk_width = mean(Spk_width, na.rm = TRUE),
                                JJ_avg = mean(JJ_avg, na.rm = TRUE),
                                JE_avg = mean(JE_avg, na.rm = TRUE),
                                JJ_med = mean(JJ_med, na.rm = TRUE),
                                JE_med = mean(JE_med, na.rm = TRUE),
                                Num_Branches = mean(Num_Branches, na.rm = TRUE) )

means1 <- as.data.frame(means1)

# rownames(means1) <- paste0(rownames(means1), '_', means1$Code)

# chart.Correlation(means1[,c(3:17)])

# Subset matrix for PCA
mat3 <- means1[,c(7:17)]
mat3 <- as.matrix(mat3)
rownames(mat3) <- rownames(means1)

# PCA
# "firebrick", "goldenrod2", "yellow"
pca2 <- prcomp(mat3, scale = TRUE)

p2 <- fviz_eig(pca2, addlabels=TRUE, 
              barfill="goldenrod2", barcolor ="firebrick",
              linecolor ="red") + ylim(0, 85) + 
  labs(y = "Explained Variance in Spike Morphology (%)", x = "Principal Component") +
  theme(panel.background = element_blank(), axis.text=element_text(size=20),
        text = element_text(size=20), panel.spacing = unit(0.1, "lines"))


pca_plot2 <- autoplot(pca2, data = means1, colour = 'PI',
                     loadings = TRUE, loadings.colour = 'blue',
                     loadings.label = TRUE, loadings.label.size = 3)

grid.arrange(p2, pca_plot2, nrow=2)



# Principal Components Regression
#https://learnche.org/pid/latent-variable-modelling/principal-components-regression
df6 <- as.data.frame(pca2$x)
df7 <- cbind(means1[, c(1:6)], df6)
# df7$sht_spk_NOgs <- log(df7$sht_spk_NOgs)

# Correlations
# chart.Correlation(df7[,c(3,5,7:17)])

sht_spk_pca <- df7[, -c(1, 4:6)]
sht_pctg_pca <- df7[, -c(1, 3:4,6)]


sht_spk_M1 <- lm(log(sht_spk_NOgs) ~ . - PI, data = sht_spk_pca)
sht_pctg_M1 <- lm(sht_pctg_NOgs ~ . - PI, data = sht_pctg_pca)


summary(sht_spk_M1)
summary(sht_pctg_M1)




# Variable selection
library(olsrr)

# sht_spk_M1 <- ols_step_all_possible(sht_spk_M1)
# sht_pctg_M1 <- ols_step_all_possible(sht_pctg_M1)

# plot(sht_spk_M1)

# Best_sht_spk_M1 <- ols_step_best_subset(sht_spk_M1)
# Best_sht_pctg_M1 <- ols_step_best_subset(sht_pctg_M1)




library(MASS)
stepAIC(sht_spk_M1, direction = 'both')
stepAIC(sht_pctg_M1, direction = 'both')

Best_sht_spk_M1 <- lm(formula = log(sht_spk_NOgs) ~ PC1 + PC2 + PC3 + PC4 + PC5 + 
                        PC7 + PC8 + PC11 + PI, data = sht_spk_pca)

Best_sht_pctg_M1 <- lm(formula = sht_pctg_NOgs ~ PC1 + PC2 + PC4 + PC5 + PC6 + PC7 + 
                         PC8 + PC9 + PC11 + PI, data = sht_pctg_pca)

summary(Best_sht_spk_M1)
anova(sht_spk_M1, Best_sht_spk_M1)

summary(Best_sht_pctg_M1)
anova(sht_pctg_M1, Best_sht_pctg_M1)






sht_spk_pca$sht_spk_NOgs <- log(sht_spk_pca$sht_spk_NOgs)
# sht_pctg_pca$sht_pctg_NOgs < log(sht_pctg_pca$sht_pctg_NOgs)

# Correlations
chart.Correlation(sht_spk_pca[,-1])
chart.Correlation(sht_pctg_pca[,c(2:8)])

CorPlotData <- sht_pctg_pca[, c(1:4,8)]
colnames(CorPlotData)[2] <- 'Shattering'
CorPlotData$PI <- as.factor(CorPlotData$PI)
chart.Correlation(CorPlotData[,-1])

# Best PCs for sht_spk (8,3,2,1)
pca_plot <- autoplot(pca2, data = means1, x = 8, y = 3, colour = 'PI',
                     loadings = TRUE, loadings.colour = 'orange',
                     loadings.label = TRUE, loadings.label.size = 4.5, repel = TRUE) +
  theme(text = element_text(size=20), axis.text.x=element_blank())

# Best PCs for sht_pctg (1, 2, 6)
pca_plot <- autoplot(pca2, data = means1, x = 1, y = 2, colour = 'PI',
                     loadings = TRUE, loadings.colour = 'orange',
                     loadings.label = TRUE, loadings.label.size = 4.5, repel = TRUE) +
  theme(text = element_text(size=20), axis.text.x=element_blank())




fviz_pca_var(res.pca, col.var = "cos2",
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
             repel = TRUE # Avoid text overlapping
)






res.pca <- prcomp(decathlon2.active, scale = TRUE)
pca2 <- prcomp(mat3, scale = TRUE)

fviz_pca_var(pca2, axes = c(2, 1),
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
) + theme(text = element_text(size=20), axis.text.x=element_blank())
