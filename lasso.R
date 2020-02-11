library(leaps)
library(glmnet)
library('Matrix')
library(ggfortify)
library(xlsx)


setwd("~/Documents/Fall 2019/Machine Learning/project/data")
load("MLProjectData.Rdata")
set.seed(7515)

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

train_no_target2 <- subset(train, select = -c(target2))
valid_no_target2 <- subset(valid, select = -c(target2))

#check correlation
X=model.matrix(target1 ~ . , data=train_no_target2 )[,-1]
y = train_no_target2 $target1

Xtest=model.matrix(target1 ~ . , data=valid_no_target2 )[,-1]
ytest = valid_no_target2 $target1

cor(X)

#ridge
cv.out.ridge=cv.glmnet(X,y,alpha=0,family="binomial")
plot(cv.out.ridge)

#' From here, we can grab the best lambda and check the model on our test data.
bestlambdaridge=cv.out.ridge$lambda.1se
(ridge.mod.betas = coef(cv.out.ridge, s=bestlambdaridge))
pred.ridge = predict(cv.out.ridge, s=bestlambdaridge, newx=Xtest)
(val.MSE.ridge = mean((pred.ridge-ytest)^2))

#lasso
cv.out.lasso=cv.glmnet(X,y,alpha=1,family="binomial")
plot(cv.out.lasso)

#' From here, we can grab the best lambda and check the model on our test data.
bestlambdalasso=cv.out.lasso$lambda.1se
(lasso.mod.betas = coef(cv.out.lasso, s=bestlambdalasso))
pred.lasso = predict(cv.out.lasso, s=bestlambdalasso, newx=Xtest)
(val.MSE.lasso = mean((pred.lasso-ytest)^2))

coeff <- as.data.frame(summary(lasso.mod.betas))
write.xlsx(coeff, "lasso_betas.xlsx")

#' Results comparable to the other methods. If we want to go with this method,
#' we'd just update the coefficients (by running the entire optimization again)
#' on the entire dataset.

out=glmnet(X,y,alpha=1,lambda=bestlambda)
lasso.coef=predict(out, type="coefficients")
lasso.coef

#' Notice the sparsity in the parameter estimates - this is the benefit of LASSO over 
#' Ridge Regression.

# Last thing to try if time permits is the ELASTIC NET
set.seed(1)
cv.out=cv.glmnet(X[train,],y[train],alpha=0.5)
plot(cv.out)

bestlambda=cv.out$lambda.min
(elastic.mod.betas = coef(cv.out, s=bestlambda))
pred.elastic = predict(cv.out, s=bestlambda, newx=X[test,])
(val.MSE.elastic = mean((pred.elastic-y[test])^2))
