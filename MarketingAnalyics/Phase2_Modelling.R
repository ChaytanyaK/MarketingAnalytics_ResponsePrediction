
#myfile3<-"C:/Users/3001777/OneDrive - LTI/Important Documents/Data Science/Sankhya/Project/Phase 3"
#setwd(myfile3)
#getwd()
#load("NewwspacePhase3.RData")

DataSet_final <- read_csv("~/Documents/STAT_615_Regression/Regression_ Project/DataSet_final.csv")


library(tidyverse)
library(broom)
library(GGally)
library(onewaytests)
library(lbutils)
#DataSet_final <- read.csv(file.choose(),header = T,na.strings = c("","NA"))
summary(DataSet_final)   #NA's replaced with Zero 
dim(DataSet_final)

DataSet_final$Active3M <- as.factor(DataSet_final$Active3M)
DataSet_final$response <- as.factor(DataSet_final$response)


#Partitioning the  data into train and test data 

library(caret)

set.seed(1240)
index <- createDataPartition(DataSet_final$response,p=0.8,list = F)

Master_Train <- DataSet_final[index,]
dim(Master_Train)  #80 percent , 982: Observations 21: Variables

Master_Test<- DataSet_final[-index,]
dim(Master_Test)   # 20 percent , 245:Observations 21: variables 


#Use Train data for building the Model 


R_Model <- glm(response~email+sms+call+n_comp+loyalty+portal+rewards+nps+n_yrs+
                 Region+Sales2019+B1Sales2019+B1Contribution+Active3M+
                 BuyingFrequency2019+B1BuyingFrequency+BrandEngagement,data=Master_Train,family=binomial)
summary(R_Model)

#Removing the Insignificant variables 

Response_Model <- glm(response~email+sms+call+loyalty+rewards+
                        nps+n_yrs+B1Sales2019,data = Master_Train,family = binomial)

summary(Response_Model)

#email : more the number of email more are the chances to respond 
#sms : if we sent more sms then the probability of responding will be increased 
#call :  more the number of calls more are the chances to respond
#loyalty : If the ChannelPartner is loyal, he is likely to respond
#rewards : if the reward is redeemed then the it more likely that they will not respond 
#nps : net promoter score : more the score the probability of responding will be increased 
#Sales2019 : more amount of Sales more likely to respond 

#Model Equation : y = bo+ b1(email) + b2(sms) + b3(calls) +b4(loyalty) +b5(rewards)+
#                         b6(nps) + b7(n_yrs) +b8(Sales2019)

#Model Equation  : 3.115e+00 + 1.397e+00(email)+ 3.623e-01(sms) + 2.351e+00(calls) + 3.589e+00(loyalty) +
#                              (-5.480e+00 )(rewards) + 1.558e-01(nps)+
#                       8.141e-02 (n_yrs) + (5.171e-06 )(Sales2019)


#Global Hypothesis Testing 

#H0 = All the variables insignificant
#H1 = Atleast one of the variable is significant 

NUll_Model <- glm(response~1,data=Master_Train,family = binomial)
anova(Response_Model,NUll_Model,test="Chisq") 


#P-value < 0.05 Reject Ho : Hence it proves that at least one of the variable is significant.


#Checking for Multi-colinearity of the variables 

library(car)
vif(Response_Model)


#Problem of Multi colinearity does exist, hence removing rewards 

Response_Model <- glm(formula = response ~ email + loyalty + call + nps + 
                        B1Sales2019 + sms + n_yrs, family = binomial, data = Master_Train)

summary(Response_Model)

#Calculate Area under the curve , ROCR for traindata 

library(ROCR)
Predprob_Response <- round(fitted(Response_Model),2)
head(Predprob_Response)

Prediction_Response <- prediction(Predprob_Response,Master_Train$response)
Performance_Response <- performance(Prediction_Response,"tpr","fpr")
plot(Performance_Response)
abline(0,1)


Auc_Response <- performance(Prediction_Response,"auc")
Auc_Response@y.values  # [1] 0.7190153



#Deciding the Cut-off Value based on the Confusion Matrix :

PredY0.3 <- ifelse(Predprob_Response>0.3,1,0)
PredY0.35 <- ifelse(Predprob_Response>0.35,1,0)
PredY0.4 <-ifelse(Predprob_Response>0.4,1,0)
PredY0.45<-ifelse(Predprob_Response>0.45,1,0)
PredY0.47<-ifelse(Predprob_Response>0.47,1,0)

PredY0.3 <- as.factor(PredY0.3)
PredY0.35 <- as.factor(PredY0.35)
PredY0.4 <- as.factor(PredY0.4)
PredY0.45 <-as.factor(PredY0.45)
PredY0.47 <-as.factor(PredY0.47)

Master_Train$response <- as.factor(Master_Train$response)

confusionMatrix(PredY0.3,Master_Train$response,positive = '1')  #BA:73

confusionMatrix(PredY0.35,Master_Train$response,positive="1") 

confusionMatrix(PredY0.4,Master_Train$response,positive="1")     #BA:74

confusionMatrix(PredY0.45,Master_Train$response,positive="1")

#0.35 Cut-off turns out to be the best as the balanced accuracy is 76% highest and Accuracy of the Model is 79%

Master_Train$Y <- ifelse(Predprob_Response>0.4,1,0)



#Validating the Model on the Test Data on Test data 

Predprob_Test <- round(predict(Response_Model,Master_Test,type='response'),2)
head(Predprob_Test)

Prediction_Test <- prediction(Predprob_Test,Master_Test$response)
Performance_Test <- performance(Prediction_Test,"tpr","fpr")
plot(Performance_Test)
abline(0,1)

Auc_Test <- performance(Prediction_Test,"auc")
Auc_Test@y.values    #[1] 0.703561



Master_Test$PredY <- ifelse(Predprob_Test>0.4,1,0)
Master_Test$PredY <- as.factor(Master_Test$PredY)
confusionMatrix(Master_Test$PredY,Master_Test$response,positive="1")  #BA :75  Model Accuracy : 78



#K-fold Validation :

library(caret)

kfolds <- trainControl(method="cv",number = 4)

KModel <- train(as.factor(response)~email+sms+call+loyalty+rewards+
                  nps+n_yrs+B1Sales2019,method="glm",data = DataSet_final,family = binomial,trControl=kfolds)

KModel

PredprobYK <- round(fitted(KModel),2)
PredYK <- ifelse(PredprobYK>0.5,1,0)
PredYK <- as.factor(PredYK)

confusionMatrix(PredYK,DataSet_final$response,positive = "1")    #Accuracy :79  BA: 76 


#Stepwise Forward Method to align the Variables 

NUll_Model <- glm(response~1,DataSet_final,family = binomial)
Full <-glm(response~email+sms+call+n_comp+loyalty+portal+rewards+nps+n_yrs+
             Region+Sales2019+B1Sales2019+B1Contribution+Active3M+
             BuyingFrequency2019+B1BuyingFrequency+BrandEngagement,data = DataSet_final,family = binomial)

step(NUll_Model,scope=list(lower=NUll_Model,upper=Full),direction="forward")


FinalModelBLR <- glm(formula = response ~ email + loyalty + call + nps + 
                       B1Sales2019 + sms + n_yrs, family = binomial, data = DataSet_final)



summary(FinalModelBLR)


library(ROCR)
Predprob_Final_Response <- round(fitted(FinalModelBLR),2)
head(Predprob_Final_Response)

Prediction_Final_Response <- prediction(Predprob_Final_Response,DataSet_final$response)
Performance_Final_Response <- performance(Prediction_Final_Response,"tpr","fpr")
plot(Performance_Final_Response)
abline(0,1)


Auc_Final_Response <- performance(Prediction_Final_Response,"auc")
Auc_Final_Response@y.values   #0.7157219









save.image("NewwspacePhase3.RData")
