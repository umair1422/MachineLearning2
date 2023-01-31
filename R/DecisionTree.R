#Import data 2.1
bank_data<-read.csv("bank-full.csv", sep = ";")

bank_data[,17] <- as.factor(bank_data[,17])
bank_data[,1] <- as.numeric(bank_data[,1])
bank_data[,2] <- as.factor(bank_data[,2])
bank_data[,3] <- as.factor(bank_data[,3])
bank_data[,4] <- as.factor(bank_data[,4])
bank_data[,5] <- as.factor(bank_data[,5])
bank_data[,6] <- as.numeric(bank_data[,6])
bank_data[,7] <- as.factor(bank_data[,7])
bank_data[,8] <- as.factor(bank_data[,8])
bank_data[,9] <- as.factor(bank_data[,9])
bank_data[,10] <- as.factor(bank_data[,10])
bank_data[,11] <- as.factor(bank_data[,11])
bank_data[,12] <- as.numeric(bank_data[,12])
bank_data[,13] <- as.numeric(bank_data[,13])
bank_data[,14] <- as.numeric(bank_data[,14])
bank_data[,15] <- as.numeric(bank_data[,15])
bank_data[,16] <- as.factor(bank_data[,16])
bank_data<-bank_data[,-12]

n<-dim(bank_data)[1]
set.seed(12345)
id<-sample(1:n, floor(n*0.4))
train<-bank_data[id,]
id1<-setdiff(1:n, id)
set.seed(12345)
id2<-sample(id1, floor(n*0.3))
valid<-bank_data[id2,]
id3<-setdiff(id1,id2)
test<-bank_data[id3,]
# Fit decisionTree 2.2
require(tree)
#Fitting model on training data
# with default settings
model_train1<-tree(y~ .,data = train)
predict_train1 <- predict(model_train1, newdata = train, type = "class")
#node size equal to 7000.
model_train2 <- tree(y ~ ., data = train, control = tree.control(nobs = nrow(train), minsize = 7000))
predict_train2 <- predict(model_train2, newdata = train, type = "class")

#minimum deviance to 0.0005.
model_train3 <- tree(y ~ ., data = train, control = tree.control(nobs = nrow(train), mindev = 0.0005))
predict_train3 <- predict(model_train3, newdata = train, type = "class")

#Fitting model on Validation data
predict_Valid1 <- predict(model_train1, newdata = valid, type = "class")
predict_Valid2 <- predict(model_train2, newdata = valid, type = "class")
predict_Valid3 <- predict(model_train3, newdata = valid, type = "class")

#Confusion matrix of training data
c_matrix_train1 <- table(train$y, predict_train1)
c_matrix_train2 <- table(train$y, predict_train2)
c_matrix_train3 <- table(train$y, predict_train3)
#Confusion matrix of validation data

c_matrix_valid1 <- table(valid$y, predict_Valid1)
c_matrix_valid2 <- table(valid$y, predict_Valid2)
c_matrix_valid3 <- table(valid$y, predict_Valid3)

#Classification error of training data
Class_error_train1 <- (sum(c_matrix_train1) - sum(diag(c_matrix_train1))) / sum(c_matrix_train1)
Class_error_train2 <- (sum(c_matrix_train2) - sum(diag(c_matrix_train2))) / sum(c_matrix_train2)
Class_error_train3 <- (sum(c_matrix_train3) - sum(diag(c_matrix_train3))) / sum(c_matrix_train3)
#Classification error of Validation data

Class_error_valid1 <- (sum(c_matrix_valid1) - sum(diag(c_matrix_valid1))) / sum(c_matrix_valid1)
Class_error_valid2 <- (sum(c_matrix_valid2) - sum(diag(c_matrix_valid2))) / sum(c_matrix_valid2)
Class_error_valid3 <- (sum(c_matrix_valid3) - sum(diag(c_matrix_valid3))) / sum(c_matrix_valid3)

C_error_mat_train<-matrix(ncol = 3)
C_error_mat_valid<-matrix(ncol = 3)

C_error_mat_train<-rbind(Class_error_train1,Class_error_train2,Class_error_train3)
C_error_mat_valid<-rbind(Class_error_valid1,Class_error_valid2,Class_error_valid3)
print(C_error_mat_train)
print(C_error_mat_valid)

#optimal node value

train_tree <- list()
valid_tree <- list()
train_tree[1]<-Inf
valid_tree[1]<-Inf
for(level in 2:50) {
  Tree_levels <- prune.tree(model_train3, best = level,method = "deviance")
  pred_train <- predict(Tree_levels, newdata = train, type = "tree")

  pred_valid <- predict(Tree_levels, newdata = valid, type = "tree")
  train_tree[level] <- deviance(pred_train)
  valid_tree[level] <- deviance(pred_valid)
}
opt_mat<-matrix(ncol = 2,nrow = 2)
colnames(opt_mat)<-c("Minimum deviance","Optimal Value")
opt_mat[1,]<-c(min(unlist(train_tree)),which.min(unlist(train_tree)))
opt_mat[2,]<-c(min(unlist(valid_tree)),which.min(unlist(valid_tree)))
print(opt_mat)

plot(2:50, train_tree[2:50], type="b", col="black")
plot(2:50, valid_tree[2:50], type="b", col="blue")
# 4 test data
test_Tree_levels <- prune.tree(model_train3, best = 23,method = "deviance")
pred_test <- predict(test_Tree_levels, newdata = test, type = "class")

c_matrix_test <- table(test$y, pred_test)
print(c_matrix_test)
Accuracy_test <- (sum(diag(c_matrix_test)) / sum(c_matrix_test))*100
cat("Accuracy on test data is =", Accuracy_test,"%")
percision<-(c_matrix_test[1,1]/(c_matrix_test[1,1]+c_matrix_test[2,1]))
recal<-(c_matrix_test[1,1]/(c_matrix_test[1,1]+c_matrix_test[1,2]))
F1_score<-2*((percision*recal)/(percision+recal))
cat("F1_score on test data is =", F1_score)

#5
library(rpart)
loss_matr <- matrix(c(0, 1, 5, 0), nrow = 2, byrow = TRUE)

fit_loss <- rpart(y ~ ., data = train, method = "class", parms = list(loss = loss_matr))
pred_loss <- predict(fit_loss, newdata = test, type = "class")
c_matrix_loss <- table(test$y, pred_loss)
print(c_matrix_loss)
Accuracy_loss <- (sum(diag(c_matrix_loss)) / sum(c_matrix_loss))*100
cat("Accuracy on test data with loss matrix is =", Accuracy_loss,"%")

#6
library(ROCR)
library(ggplot2)
p<-seq(0.05,0.95,by=0.05)
test_opt <- prune.tree(model_train3, best = 24,method = "deviance")
tpr_dc<-list()
fpr_dc<-list()
dc_df<-data.frame()
for(i in 1:15)
{
  pred_opt <- predict(test_opt, newdata = test, type = "vector")
  predict_thresh <- ifelse(pred_opt[,2] >p[i], "yes", "no")
  tb_dc<-table(predict_thresh,test$y)
  print(i)
  print(tb_dc)
  tpr_dc[i]<-tb_dc[2,2]/(tb_dc[2,1]+tb_dc[2,2])
  fpr_dc[i]<-tb_dc[1,2]/(tb_dc[1,1]+tb_dc[1,2])
}
dc_df = data.frame(fpr=unlist(fpr_dc),tpr=unlist(tpr_dc))
plot(dc_df,col="black")
lines(dc_df,col="black")
logistic_model <- glm(y ~ .,data = train,family = "binomial")
fpr_l<-list()
tpr_l<-list()
log_df<-data.frame()
for(i in 1:length(p))
{
  predict_reg <- predict(logistic_model, test , type = "response")
  predict_reg <- ifelse(predict_reg >p[i], "yes", "no")
  tb<-table(predict_reg,test$y)
  tpr_l[i]<-tb[2,2]/(tb[2,1]+tb[2,2])
  fpr_l[i]<-tb[1,2]/(tb[1,1]+tb[1,2])
}

log_df = data.frame(fpr=unlist(fpr_l),tpr=unlist(tpr_l))
lines(log_df,col="red")
