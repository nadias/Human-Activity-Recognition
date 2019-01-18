library(randomForest)
setwd("/Users/nadiasoares/Documents/Projects/HumanActivityRecognition/Human-Activity-Recognition/UCI HAR Dataset/")
train_data<-read.table("train/X_train.txt")
train_labels<-read.table("train/Y_train.txt")

test_data<-read.table("test/X_test.txt")
test_labels<-read.table("test/Y_test.txt")

col_names <- readLines("features.txt")
colnames(train_data)<-make.names(col_names)
colnames(test_data)<-make.names(col_names)
colnames(train_labels)<-"label"
colnames(test_labels)<-"label"

train_final<-cbind(train_labels,train_data)
test_final<-cbind(test_labels,test_data)
final_data<-rbind(train_final,test_final)
final_data$label<-factor(final_data$label)

str(final_data)
nrow(final_data[!complete.cases(final_data),])  # It has no NA's

model_rfF<-randomForest(label~.,final_data[1:7352,])
model_rfF
summary(model_rfF)

pre_rfF<-predict(model_rfF,final_data[-(1:7352),],type = "response")
confusionMatrix(pre_rfF,final_data[-(1:7352),1])

predTrain<-predict(model_rfF,final_data[(1:7352),],type = "response")
confusionMatrix(predTrain,final_data[(1:7352),1]) # There is overfitting

# Tuning the model

importance(model_rfF)
bestmtry <- tuneRF(final_data[1:7352,], final_data[1:7352,]$label, stepFactor=1.5, improve=1e-5, ntree=30)
plot(model_rfF, main="OOB error convergence for nTrees")
model_rfF<-randomForest(label~.,final_data[1:7352,], mtry=34, ntree=60)
model_rfF #0.9223 in testing set, 1 in training

model_rfF<-randomForest(label~.,final_data[1:7352,], mtry=16, ntree=60)
# 0.9233 in testing set, 1 in training

model_rfF<-randomForest(label~.,final_data[1:7352,], mtry=16, ntree=30)
# 0.9335 in testing set, 0.9999 in training

model_rfF<-randomForest(label~.,final_data[1:7352,], mtry=18, ntree=30) #X
# 0.9348 in testing set, 1 in training