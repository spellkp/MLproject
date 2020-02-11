library('randomForest')
library('dplyr')

setwd("~/Documents/Fall 2019/Machine Learning/project/data")
load("MLProjectData.Rdata")

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

train_no_target2_sampled <- sample(c(T,F), nrow(train_no_target2), rep=TRUE, p=c(0.15,0.85))
test=!train

rf2 <- randomForest(as.factor(target1) ~ ., data=train_no_target2, importance = T, ntree=20, type='class')

#' We can then examine the confusion matrix to see how our model predicts each class:

rf$confusion
rf2$confusion
#' The classwise error rates are extremely small for the 
#' entire model! \blue{rf\$err.rate} will show the progression 
#' of misclassification rates as each individual tree is added 
#' to the forest for each class and overall (on the out of bag 
#' (OOB) data that was remaining after the data was sampled to 
#' train the tree).

#rf$err.rate
head(rf2$err.rate)
rf$err.rate[50,]

#' Tells us which variables are important in prediction of each digit
importance(rf2)
library(xlsx)
importance <- as.data.frame(importance(rf2))
write.xlsx(importance, "random_forest_variable_importance.xlsx")
e