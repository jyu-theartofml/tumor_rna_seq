library(data.table)
library(plyr)
library(caret)
library(nsprcomp)
library(xgboost)

########################### Load data ##############
raw.data<-fread('data.csv')
label<-fread('labels.csv')
target<-label$Class

target.num<-mapvalues(target, from = c("BRCA","COAD", "KIRC",'LUAD','PRAD'), to = c(0,1,2,3,4))
raw.data$V1<-NULL
raw.data<-data.frame(raw.data)
df<-raw.data[,-(which(colSums(raw.data) == 0))]

############ visualize the PCA plot on whole dataset ##############
pca<-nsprcomp(df, k=c(1000,1000,1000,1000,1000,1000,1000,1000,1000,1000))
df_out <- as.data.frame(pca$x)
df_out$group <- label$Class

library(ggplot2)
library(plotly)
p <- plot_ly(df_out, x = ~PC1, y = ~PC2, z = ~PC3, color = ~group) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'PC1'),
                      yaxis = list(title = 'PC2'),
                      zaxis = list(title = 'PC3')))
p
########### perform PCA on training data for predictive modeling #################

cv_folds <- createFolds(target, k=5)
val_split<-lapply(cv_folds, function(ind, dat) dat[ind,], dat=df)
train_split<-lapply(cv_folds, function(ind, dat) dat[-ind,], dat=df)
#for 5 CV and 5 classes F1 scores
f1_val<-matrix(NA, nrow=5, ncol=5)

for (i in 1:length(cv_folds)){
  group<-names(cv_folds)[i]
  #get dataframe using [[]]
  val.df<-val_split[[group]]
  train.df<-train_split[[group]]
  #perform sparse PCA on training set, limit to 1K non-zero loadings for the feature variables (5% of total features)
  library(doSNOW)
  library(ff)
  library(doSNOW)
  cluster<-makeCluster(6,"SOCK")
  registerDoSNOW(cluster)
  spca.train<-nsprcomp(train.df, k=c(1000,1000,1000,1000,1000,1000,1000,1000,1000,1000))
  train.spc<-spca.train$x
  val.spc<-predict(spca.train, val.df)
  
  #get target info
  valIdx<-cv_folds[group][[group]]
  train.target<-target.num[-valIdx]
  val.target<-target.num[valIdx]
  
  #set up for xgb
  dtrain.sparse<-xgb.DMatrix(data = as.matrix(train.spc), label =as.numeric(train.target))
  dval.sparse<-xgb.DMatrix(data = as.matrix(val.spc), label = as.numeric(val.target))
  watchlist <- list(train=dtrain.sparse, test=dval.sparse)
  xgb.model <- xgb.train(data=dtrain.sparse, max_depth=2, eta=0.0001, nthread = 2,colsample_bytree=0.3, nrounds=150, verbose=F,
                         watchlist=watchlist,num_class=5, early_stopping_rounds=3,objective = "multi:softmax", eval_metric="merror")
  
  prob<-predict(xgb.model,dval.sparse)
  # sebastian raschka:What we are trying to achieve with the F1-score metric is to find an equal balance between precision and recall, which is extremely useful in most scenarios when we are working with imbalanced datasets (i.e., a dataset with a non-uniform distribution of class labels).
  g<-confusionMatrix(prob, val.target)
  f1_val[i,]<-as.vector(g$byClass[,7])
}
  
colnames(f1_val) <- c("BRCA","COAD", "KIRC",'LUAD','PRAD')
rownames(f1_val)<-c("CV1","CV2", "CV3",'CV4','CV5')
as.table(f1_val)
