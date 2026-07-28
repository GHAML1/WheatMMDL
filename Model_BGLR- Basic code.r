# %%
rm(list=ls(all=TRUE))
library(BGLR); library(dplyr); library(writexl)
library(ggplot2); library(gridExtra)
library(readxl)#library(DescTools)#AUC
library(tidyr)
type = 'RIL'
# type = 'variety'
load(paste0('data-PP/data-PP-',type,'.RData'), verbose=TRUE)
head(pheno)
row.names(weather_smms) =  weather_smms$year


summary(abs(weather_smms[3,]-weather_smms[2,]))

#Kernel-G
library(MASS)
KG_f<-function(X=NULL,Geno=NULL,q=0.5,h=NULL){
  #dist2_X = dist(X)
  #h = quantile(dist2_X[lower.tri(dist2_X)],0.5)
  #exp(-2*dist_X/h)
  if(!is.null(Geno)){
    Squared_Norms = matrix(diag(Geno), nrow(Geno), nrow(Geno))
    Dist2_X = Squared_Norms+t(Squared_Norms) - 2 * Geno
    Dist2_X =  Dist2_X/max(Dist2_X[lower.tri(Dist2_X)])##Dist2_X =  Dist2_X/max(c(Dist2_X[lower.tri(Dist2_X)],0.0001))
    if(is.null(h)) h = quantile(Dist2_X[lower.tri(Dist2_X)],q)
  }else{
    XtX = tcrossprod(X)
    Squared_Norms = matrix(diag(XtX), nrow(X), nrow(X))
    Dist2_X = Squared_Norms+t(Squared_Norms) - 2 * XtX
    h = quantile(Dist2_X[lower.tri(Dist2_X)],q)  
  }
  exp(-2*Dist2_X/h)
  #max(abs(dist2_X^2-Dist2_X))
}
# %%
#Predictions in each year
K =  10
years =  unique(pheno$year)
traits = c('gpc','tw','gy_norm','gyd','gpd')
# traits = sub('BLUE_','',traits)
traits

## Fixed example
year <- years[1]
trait <- "gpc"
predictor <- "GID+NIRs"

# Phenotypes
pheno_r <- droplevels(pheno[pheno$year == year, ])
pheno_r <- droplevels(pheno_r[!is.na(pheno_r[[trait]]), ])

# Predictors
nirs_r <- nirs[nirs$year == year, ]
nirs_r <- nirs_r[match(pheno_r$genotype, nirs_r$genotype), ]

drone_r <- df_smm_drones[
  match(paste(pheno_r$year, pheno_r$genotype, sep = "_"),
        paste(df_smm_drones$year, df_smm_drones$genotype, sep = "_")), ]

# Gaussian kernels
Ker_markers <- KG_f(X = scale(markers[pheno_r$genotype, ]))
Ker_nirs    <- KG_f(X = scale(nirs_r[, -(1:3)]))
Ker_drone   <- KG_f(X = scale(drone_r[, -(1:3)]))

# Model components
ETA <- list(
  GID   = list(K = Ker_markers, model = "RKHS"),
  NIRs  = list(K = Ker_nirs,    model = "RKHS"),
  Drone = list(K = Ker_drone,   model = "RKHS")
)

ETA_t <- ETA[strsplit(predictor, "+", fixed = TRUE)[[1]]]

# One cross-validation split
set.seed(123)
K <- 10
folds <- cut(sample(nrow(pheno_r)), breaks = K, labels = FALSE)

k <- 1
tst_idx <- which(folds == k)

y_na <- pheno_r[[trait]]
y_na[tst_idx] <- NA

FM <- BGLR(
  y = y_na,
  ETA = ETA_t,
  nIter = 40000,
  burnIn = 20000,
  verbose = FALSE
)

predictions <- data.frame(
  genotype = pheno_r$genotype[tst_idx],
  observed = pheno_r[[trait]][tst_idx],
  predicted = FM$yHat[tst_idx]
)

head(predictions)