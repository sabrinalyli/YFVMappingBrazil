##Compute 95%CI Uncertainties using boostrap samples using SuperLearner
##Rerun code for each individual species 

# set up a cluster of four cpus in with parallel execution.
sfInit (parallel=TRUE , cpus=25)

sfLibrary(snowfall)
sfLibrary(SuperLearner)
sfLibrary(seegSDM)
sfLibrary(raster)
sfLibrary(sf)
sfLibrary(glmnet)
sfLibrary(xgboost)
sfLibrary(ranger)
sfLibrary(purrr)
sfLibrary(tidyverse)

#import df containing presence/absence records for species  and environmental covariates
occ_covariates_df<-readRDS("occ_hg_jan_geo_post1990.rds") %>%
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

#use covariates rasterstack as environmental data for prediction
x_new<-readRDS("covariates_stack_df.rds") %>%
  dplyr::select(-x,-y)

#bootstrap sampling the dataset with replacement, which means that the same observation can be sampled
#more than once, but the returning sample will have the same # of observations as the original sample. Use lapply afterwards so that a list can be returned for each sample 

sub <- function(i, dat) {
  subsample(dat,
            nrow(dat),
#ensuring we have 100 presence and absence records 
            min = c(100,100), 
            replace = TRUE,
#column number for the presence/absence code (containing 1s for presences and 0s for absences)
            prescol= 1) 
}

#create 100 lists 
data_list <- lapply(1:100,  
                    sub,
                    occ_covariates_df_noNA)

#run a model on each dataset
#sfLapply which acts like lapply, except that each element of the list is processed in parallel.

set.seed(1)
model_list <- sfLapply(data_list,
                     function (x)
                       SuperLearner(
                         Y= x[,1],
                         X= x[,2:15],
                         family = binomial(),
                         cvControl = list(V = 10),
                         #cluster = cluster,
                         SL.library = c("SL.glmnet","SL.xgboost","SL.ranger")) 
)
sfExport('model_list')

# split x_new into chunks of 1000-row dataframes
## A subset below for cross-checking
# x_new_small<-x_new[sample(nrow(x_new), 1000), ]
# x_new_split_subset<- split(x_new_small[1:21,], (seq(nrow(x_new_small[1:21,]))-1) %/% 10) 

x_new_split<- split(x_new, (seq(nrow(x_new))-1) %/% 1000) 

#combines model predictions from preds object into columns of a dataframe 
combine_model_predictions <- function(preds) {
  preds_combined <- data.frame()[1:length(preds[[1]]$pred), ] 
#initialise empty df with nrow as the number of rows of 1st model's predictions
  for (model_number in 1:length(preds)) { # iteratively append preds
    preds_combined <- cbind(preds_combined, preds[[model_number]]$pred)
  }
  return(preds_combined)
}

# computes 95% quantile interval length for model predictions
compute_quantile_diff <- function(prediction)
{return(quantile(prediction, probs=c(.975)) - quantile(prediction, probs=c(.025)))}

# computes 95% quantile interval length for model predictions of x_new
compute_uncertainty <- function(x_new_chunk) {
  # generate predictions for each model
  preds <- sfLapply(model_list, function(x) predict(x, x_new_chunk)) 
  # combine predictions into a dataframe
  preds_combined <- combine_model_predictions(preds) 
  # MARGIN=1 means apply function along rows
  # calculate 95% quantile interval length for each set of model    predictions
  uncertainty <- sfApply(x=preds_combined, margin=1, fun=compute_quantile_diff) 
  return(uncertainty)
}

#Export data from parallel execution 
sfExport('combine_model_predictions','compute_quantile_diff','compute_uncertainty')

#Compute uncertainties
uncertainties <- sapply(X=x_new_split, FUN=compute_uncertainty)

#Tabulate uncertainty outputs into a dataframe
uncertainties.tb<-uncertainties %>% 
  map_df(as_tibble) %>%
  rename(uncertainties=value)
uncertainties.df<-as.data.frame(uncertainties.tb)

#====merge uncertainties with raster file for mapping=====

#import coordinates 
coords<-readRDS("covariates_stack_df.rds") %>%
  dplyr::select(x,y)

#combine coordinates with uncertainty estimates
coords<-cbind(coords,uncertainties.df$uncertainties)

#import raster to extract extent (any covariate raster with the same extent will do) 
ras<-raster("covariates/elev_min.tif")

#convert coordinates to points 
pts<-coords
coordinates(pts) <- ~x+y

#convert location coordinates to raster 
dfr<- lapply(uncertainties.df, function(x) rasterize(pts, ras, 
                                                     uncertainties.df$uncertainties))

#====write data to raster====
writeRaster(dfr$uncertainties, "hgjan_uncertainties_ras.tif", format="GTiff")
#,overwrite=TRUE)
saveRDS(dfr$uncertainties, "hgjan_uncertainties_df_ras.rds")

#Now we've finished with the parallel cluster, we should shut it down
sfStop()
