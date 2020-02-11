library('randomForest')
library('Matrix')
library('xgboost')
library('xtable')
library('dplyr')
library('xlsx')

load("C:/fall3/machine_learning/MLProjectData.RData")

#clean data
#impute missing values in train (no missing in valid)
train$L6[is.na(train$L6)] = mean(train$L6, na.rm=TRUE)
train$M6[is.na(train$M6)] = mean(train$M6, na.rm=TRUE)
train$N6[is.na(train$N6)] = mean(train$N6, na.rm=TRUE)
train$T6[is.na(train$T6)] = mean(train$T6, na.rm=TRUE)
train$U6[is.na(train$U6)] = mean(train$U6, na.rm=TRUE)
train$V6[is.na(train$V6)] = mean(train$V6, na.rm=TRUE)
train$W6[is.na(train$W6)] = mean(train$W6, na.rm=TRUE)
train$X6[is.na(train$X6)] = mean(train$X6, na.rm=TRUE)

#transform non-numeric variables to numeric
train <- transform(train, Y2 = as.numeric(Y2), Z2 = as.numeric(Z2))
valid <- transform(valid, Y2 = as.numeric(Y2), Z2 = as.numeric(Z2))

train_sub <- train[1:10000,]
valid_sub <- valid[1:2000,]

# Training PCA components
ml_pca <- princomp(train[1:148])
target1 <- train$target1
pc1 <- ml_pca$scores[,1]
pc2 <- ml_pca$scores[,2]
prin_df <- data.frame(target1, pc1, pc2)

# View loadings of PC1 and PC2
l <- xtable(unclass(ml_pca$loadings))
loadings <- select(l, Comp.1,Comp.2)
write.xlsx(loadings, "C:/fall3/machine_learning/loadings.xlsx")

# Validation PCA components
ml_pca_v <- princomp(valid[1:148])
target1_v <- valid$target1
pc1_v <- ml_pca_v$scores[,1]
pc2_v <- ml_pca_v$scores[,2]
prin_df_v <- data.frame(target1_v, pc1_v, pc2_v)
colnames(prin_df_v) <- c("target1","pc1","pc2")

# Random forest using PCA
rf <- randomForest(factor(target1) ~ pc1 + pc2, data=prin_df, ntree=50, type="class")
rf$confusion
rf$err.rate[50,]

# Gradient boosting
xtrain <- model.matrix(factor(target1) ~ pc1 + pc2, data=prin_df)
xtest <- model.matrix(factor(target1) ~ pc1 + pc2, data=prin_df_v)
ytrain <- as.numeric(levels(factor(prin_df$target1)))[factor(prin_df$target1)]
ytest <- as.numeric(levels(factor(prin_df_v$target1)))[factor(prin_df_v$target1)]

ytrain2 <- prin_df$target1

set.seed(7515)
xgb <- xgboost(data=xtrain, 
        label=ytrain,
        eta = 0.05,
        max_depth=15,
        gamma=0,
        nround=100,
        subsample=0.75,
        colsample_bylevel=0.75,
        num_class=10,
        objective="multi:softmax",
        nthread=3,
        eval_metric='merror',
        verbose=0)

xgb <- xgboost(data=xtrain, 
               label=ytrain2,
               eta = 0.05,
               max_depth=15,
               gamma=0,
               nround=100,
               subsample=0.75,
               colsample_bylevel=0.75,
               num_class=2,
               objective="multi:softmax",
               nthread=3,
               eval_metric='merror',
               verbose=0)

ptrain <- predict(xgb, xtrain)
pvalid <- predict(xgb, xtest)

cat('Confusion Matrix:')
table(pvalid,valid$target1)

table(prin_df_v$target1)
XGBmrt <- sum(ptrain!=train$target1)/length(train$target1)
XGBmrv <- sum(pvalid!=valid$target1)/length(valid$target1)
cat('XGB Training Misclassification Rate:', XGBmrt)
cat('XGB Validation Misclassification Rate:', XGBmrv)
