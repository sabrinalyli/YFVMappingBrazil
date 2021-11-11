#calculate AUC on holdout data 

#install.packages("SuperLearner")
library(SuperLearner)
library(ck37r)
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
library(tidyverse)

#import data
occ_covariates_df1<-readRDS("occ/occ_hg_jan_geo_post1990.rds") %>%
  dplyr::select(-Year,-landc12ext,-wsf_2015_brazil,
                -ntempavg,-ntempmin,-ntempmax, 
                -lstempmin,-lstempmax,-lstempavg,
                -landc11ext,-landc08ext,
                -forest_gain_2012, 
                -slope_med,
                -Admin,-GAUL,-long,-lat,-optional) %>%
  mutate(id=1:nrow(.))

occ_covariates_df2<-readRDS("occ/occ_hg_leuco_geo_post1990_3.rds") %>%
  dplyr::select(-Year,-landc12ext,-wsf_2015_brazil,
                -ntempavg,-ntempmin,-ntempmax, 
                -lstempmin,-lstempmax,-lstempavg,
                -landc11ext,-landc08ext,
                -forest_gain_2012, 
                -slope_med,
                -Admin,-GAUL,-long,-lat,-optional) %>%
  mutate(id=1:nrow(.))

occ_covariates_df3<-readRDS("occ/occ_sa_confirmed_geo_post1990.rds") %>%
  dplyr::select(-Year,-landc12ext,-wsf_2015_brazil,
                -ntempavg,-ntempmin,-ntempmax, 
                -lstempmin,-lstempmax,-lstempavg,
                -landc11ext,-landc08ext,
                -forest_gain_2012, 
                -slope_med,
                -Admin,-GAUL,-long,-lat,-optional) %>%
  mutate(id=1:nrow(.))

occ_covariates_df1_noNA<-occ_covariates_df1 %>% 
  tidyr::drop_na(2:15)

occ_covariates_df2_noNA<-occ_covariates_df2 %>% 
  tidyr::drop_na(2:15)

occ_covariates_df3_noNA<-occ_covariates_df3 %>% 
  tidyr::drop_na(2:15)

outcome1<-occ_covariates_df1_noNA$presence
outcome2<-occ_covariates_df2_noNA$presence
outcome3<-occ_covariates_df3_noNA$presence

cov1<-occ_covariates_df1_noNA[,2:15]
cov2<-occ_covariates_df2_noNA[,2:15]
cov3<-occ_covariates_df3_noNA[,2:15]

#use covariates as new data for prediction
x_new<-readRDS("covariates_stack_df.rds") %>%
  dplyr::select(-x,-y)

#create training data
# Set a seed for reproducibility in this random sampling.
set.seed(1)

# Reduce to a dataset containing 80% of observations
train_obs1 = sample(nrow(cov1), 955) #hg_jan
train_obs2 = sample(nrow(cov2),1255) #hg_leuco
train_obs3 = sample(nrow(cov3),1114) #sa

# X is our training sample.
x_train1 = cov1[train_obs1, ]
x_train2 = cov2[train_obs2, ]
x_train3 = cov3[train_obs3, ]

# Create a holdout set for evaluating model performance.
# Note: cross-validation is even better than a single holdout sample.
x_holdout1 = cov1[-train_obs1, ]
x_holdout2 = cov2[-train_obs2, ]
x_holdout3 = cov3[-train_obs3, ]

#set outcome training data 
y_train1 = outcome1[train_obs1]
y_train2 = outcome2[train_obs2]
y_train3 = outcome3[train_obs3]

y_holdout1 = outcome1[-train_obs1]
y_holdout2 = outcome2[-train_obs2]
y_holdout3 = outcome3[-train_obs3]

# Review the outcome variable distribution.
table(y_train, useNA = "ifany")

#run model on training data 
model_train1 = SuperLearner(Y = y_train1, X = x_train1,
                             family = binomial(),
                             #For a real analysis we would use V = 10.
                             cvControl = list(V = 10),
                             SL.library = c("SL.glmnet","SL.xgboost","SL.ranger"))  

model_train2 = SuperLearner(Y = y_train2, X = x_train2,
                           family = binomial(),
                           #For a real analysis we would use V = 10.
                           cvControl = list(V = 10),
                           SL.library = c("SL.glmnet","SL.xgboost","SL.ranger"))  

model_train3 = SuperLearner(Y = y_train3, X = x_train3,
                           family = binomial(),
                           #For a real analysis we would use V = 10.
                           cvControl = list(V = 10),
                           SL.library = c("SL.glmnet","SL.xgboost","SL.ranger"))  

# # Predict back on the holdout dataset.
# # onlySL is set to TRUE so we don't fit algorithms that had weight = 0, saving computation.
pred_holdout1 = predict(model_train1, x_holdout1) #onlySL = TRUE)
pred_holdout2 = predict(model_train2, x_holdout2) #onlySL = TRUE)
pred_holdout3 = predict(model_train3, x_holdout3) #onlySL = TRUE)

# Review AUC - Area Under Curve
pred_rocr1 = ROCR::prediction(pred_holdout1$pred, y_holdout1)
auc1 = ROCR::performance(pred_rocr1, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc1

pred_rocr1 = ROCR::prediction(pred_holdout1$pred, y_holdout1)
acc1 = ROCR::performance(pred_rocr1, measure = "acc", x.measure = "cutoff")@y.values[[1]]
acc1


pred_rocr2 = ROCR::prediction(pred_holdout2$pred, y_holdout2)
auc2 = ROCR::performance(pred_rocr2, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc2

pred_rocr3 = ROCR::prediction(pred_holdout3$pred, y_holdout3)
auc3 = ROCR::performance(pred_rocr3, measure = "auc", x.measure = "cutoff")@y.values[[1]]
auc3

#calculate CI of auc 
auc_ci1<-pROC::ci.auc(y_holdout1, pred_holdout1$pred) 
auc_ci1<-as.data.frame(auc_ci1)
auc_ci_comb1<-data.frame(lci=c(auc_ci1[1,]),uci=c(auc_ci1[3,]))
auc_ci_comb1

auc_ci2<-pROC::ci.auc(y_holdout2, pred_holdout2$pred) 
auc_ci2<-as.data.frame(auc_ci2)
auc_ci_comb2<-data.frame(lci=c(auc_ci2[1,]),uci=c(auc_ci2[3,]))
auc_ci_comb2

auc_ci3<-pROC::ci.auc(y_holdout3, pred_holdout3$pred) 
auc_ci3<-as.data.frame(auc_ci3)
auc_ci_comb3<-data.frame(lci=c(auc_ci3[1,]),uci=c(auc_ci3[3,]))
auc_ci_comb3

auc_ci<-rbind(auc_ci_comb1,auc_ci_comb2,auc_ci_comb3)


auc.df<-data.frame(species =c("Hg. janthinomys","Hg. leucocelaenus","Sabethes spp."), 
                   auc = c(auc1,auc2,auc3),
                   ci=auc_ci)

#plot relinf
png("stacked_results/auc_combined.png",width=5,height=5,units="in",res=300)
ggplot(auc.df, aes(x=species,y= auc)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymax = ci.lci, ymin = ci.uci),width = 0.25) +
  #expand_limits(x= c(0, 1))+ 
  #scale_y_continuous(breaks = seq(0, 50, 5)) +
  labs(y="AUC") 
  #coord_flip()
dev.off()
