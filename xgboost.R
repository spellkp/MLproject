library(xgboost)

setwd("~/Documents/Fall 2019/Machine Learning/project/data")
load("MLProjectData.Rdata")

library('Matrix')

sapply(train, function(x) sum(is.na(x)))

train$L6[is.na(train$L6)] = mean(train$L6, na.rm=TRUE)
train$M6[is.na(train$M6)] = mean(train$M6, na.rm=TRUE)
train$N6[is.na(train$N6)] = mean(train$N6, na.rm=TRUE)
train$T6[is.na(train$T6)] = mean(train$T6, na.rm=TRUE)
train$U6[is.na(train$U6)] = mean(train$U6, na.rm=TRUE)
train$V6[is.na(train$V6)] = mean(train$V6, na.rm=TRUE)
train$W6[is.na(train$W6)] = mean(train$W6, na.rm=TRUE)
train$X6[is.na(train$X6)] = mean(train$X6, na.rm=TRUE)
sapply(train, function(x) sum(is.na(x)))
sapply(train, class)
train <- transform(train, Y2 = as.numeric(Y2), Z2 = as.numeric(Z2))
valid <- transform(valid, Y2 = as.numeric(Y2), Z2 = as.numeric(Z2))

train_no_target2 <- subset(train, select = -c(target2))
valid_no_target2 <- subset(valid, select = -c(target2))
valid_no_target2 $L6_missing <- as.numeric(is.na(valid_no_target2 $L6))

train_imp_var <- select(train, U6, W6, P2, Q2, F6, E1, I4, F5, H6, T1, K3, M1, 
                        K6, N4, N5, F1,Q3, R2, S6, Y2, W2, M5, P6, R6,X6, N1, 
                        target1, target2)


valid_imp_var <- select(valid, U6, W6, P2, Q2, F6, E1, I4, F5, H6, T1, K3, M1, 
                        K6, N4, N5, F1,Q3, R2, S6, Y2, W2, M5, P6, R6,X6, N1, 
                        target1, target2)

xtrain = model.matrix(factor(target1) ~ . , data=train_no_target2)
ytrain = train_no_target2$target1 

xtest = model.matrix(factor(target1) ~ . , data=valid_no_target2)
ytest = valid_no_target2$target1 


set.seed(7515)
# run the model (assuming parameters are already tuned)
xgb <- xgboost(data = xtrain,
                   label = ytrain,
                   eta = 0.05,
                   max_depth = 15,
                   gamma = 0,
                   alpha = 10,
                   nround=100,
                   subsample = 0.75,
                   colsample_bylevel = 0.25,
                   num_class = 2,
                   objective = "multi:softmax",
                   nthread = 3,
                   eval_metric = 'merror',
                   verbose =0)

ptrain = predict(xgb, xtrain)
XGBmrt = sum(ptrain!=ytrain)/length(ytrain )
cat('XGB Training Misclassification Rate:', XGBmrt)

pvalid = predict(xgb, xtest)

cat('Confusion Matrix:')
table(pvalid,ytest)

XGBmrv = sum(pvalid!=ytest)/length(ytest)
cat('XGB Validation Misclassification Rate:', XGBmrv)

importance <- xgb.importance(feature_names = colnames(xtrain), model = xgb)
head(importance,30)
library(xlsx)
write.xlsx(importance, "xgboost_variable_importance.xlsx")
