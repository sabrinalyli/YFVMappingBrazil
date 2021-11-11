library(SuperLearner)
library(seegSDM)
library(tidyr)
library(dplyr)
library(raster)
library(sf)
library(glmnet)
library(xgboost)
library(kernlab)
library(ranger)
library(purrr)


#import df containing presence/absence records for species  and environmental covariates
occ_covariates_df<-readRDS("occ/occ_hg_jan_geo_post1990.rds") %>%
  dplyr::select(-Year,-landc12ext,-wsf_2015_brazil,
                -ntempavg,-ntempmin,-ntempmax, 
                -lstempmin,-lstempmax,-lstempavg,
                -landc11ext,-landc08ext,
                -forest_gain_2012, 
                -slope_med,
                -Admin,-GAUL,-long,-lat,-optional) %>%
  mutate(id=1:nrow(.))

#check for missing data in covariates - df cannot have missing data 
colSums(is.na(occ_covariates_df))

#remove rows with missing data for covariate columns f
occ_covariates_df_noNA<-occ_covariates_df %>% 
  tidyr::drop_na(2:15)

#set outcome variable
outcome<-occ_covariates_df_noNA$presence

#set covariates
cov<-occ_covariates_df_noNA[,2:15]

#use covariates rasterstack as new data for prediction
x_new<-readRDS("covariates_stack_df.rds") %>%
  dplyr::select(-x,-y)

#create training data
# Set a seed for reproducibility in this random sampling.
set.seed(1)

# Reduce to a dataset containing 80% of observations
train_obs = sample(nrow(cov), 955) #hg_jan
#train_obs = sample(nrow(cov2),1255) #hg_leuco
#train_obs = sample(nrow(cov3),1114) #sa
#train_obs = sample(nrow(cov),1099) #hg_leuco

#X is our training sample.
x_train = cov[train_obs, ]

#set outcome training data 
y_train = outcome[train_obs]

# Review the outcome variable distribution.
table(y_train, useNA = "ifany")

#run superlearner ensemble on training data
cv_sl_forpred = SuperLearner(Y = y_train, X = x_train,
                          family = binomial(),
                          #For a real analysis we would use V = 10.
                          cvControl = list(V = 10),
                          #cluster = cluster,
                          SL.library = c("SL.glmnet","SL.xgboost","SL.ranger"))  

saveRDS(cv_sl_forpred,"cv_sl_forpred_hgjan.rds")
cv_sl_forpred_old<-readRDS("cv_sl_forpred.rds")

# Plot the performance of models
plot(cv_sl_forpred) + theme_bw()

# Save plot to a file.
#ggsave("SuperLearner_hgjan.png")

# # Predict back on the new data 
# # onlySL is set to TRUE so we don't fit algorithms that had weight = 0, saving computation.
pred = predict(cv_sl_forpred, x_new, onlySL = TRUE)

#====create raster map output for predicted results====
# #merge predicted results with covariates
coords<-readRDS("covariates_stack_df.rds") %>%
dplyr::select(x,y)

#convert raster to df
pred.df<-as.data.frame(pred)

#link df to coordinates
coords<-cbind(coords,pred.df$pred)

#import raster for extent (any raster with the same extent will do)
ras<-raster("covariates/elev_min.tif")

#convert df to spdf 
pts<-coords
coordinates(pts) <- ~x+y

#create raster map for predicted outputs 
dfr<- lapply(pred, function(x) rasterize(pts, ras, pred$pred))
writeRaster(dfr$pred, "hgjan_pred_stack_three_test2.tif")#, format="GTiff")
