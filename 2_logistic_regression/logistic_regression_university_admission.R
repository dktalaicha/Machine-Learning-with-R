#https://www.kaggle.com/shraban020/predicting-admission-by-logistic-regression
# https://www.kaggle.com/vrishabhnaik/chance-of-admit-linear-regression-model
# https://www.kaggle.com/naishalthakkar/gre-dataset-analysis
# https://www.kaggle.com/ravichaubey1506/postgraduate-admission-analysis

#Load required library or packages
library(zoo) 
library(carData) 
library(car) 
library(lmtest) 
library(nortest)
library(dplyr) 
library(stringr)
library(summarytools)
library(caret)
library(nnet)
library(ROCR)


#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/logistic_regression/2_logistic_regression")
getwd()

#Importing the csv data
Admission <- read.csv("university_admission_discrete.csv")
Admission
View(Admission)

dim(Admission)

str(Admission)

#Cleaning the data
colnames(Admission)
colnames(Admission) <- str_replace_all(colnames(Admission),"[.]","")
colnames(Admission)

# checking duplicate rows
length(unique(Admission$Serial.No.)) == nrow(Admission)
# Serial.No.is the primary key with no duplicate values

# Checking missing values
sum(is.na(Admission))
# Total 4 missing values in dataset

colnames(Admission)
dfSummary(Admission)

Admission[is.na(Admission$GREScore),2] <- round(mean(Admission$GREScore,na.rm = TRUE))
Admission[is.na(Admission$TOEFLScore),3] <- round(mean(Admission$TOEFLScore, na.rm = TRUE))
Admission[is.na(Admission$UniversityRating),4] <- round(mean(Admission$UniversityRating, na.rm = TRUE))
Admission[is.na(Admission$CGPA),7] <- round(mean(Admission$CGPA, na.rm = TRUE))

dfSummary(Admission)
sum(is.na(Admission))


#Dataframe Summary
# dfSummary function from summarytools package
dfSummary(Admission)

# Analysing variable 'GRE.Score'
summary(Admission$GREScore)
boxplot(Admission$GREScore)

df <- Admission

summary(df$GREScore)
quantile(df$GREScore, seq(0,1,0.01))
q1 <- quantile(df$GREScore, c(0.25))
q3 <- quantile(df$GREScore, c(0.75))
IQR <- q3 - q1  
upper_range <- q3 + 1.5*IQR  
lower_range <- q1 - 1.5*IQR
nrow(df[df$GRE.Score > upper_range,])
nrow(df[df$GRE.Score < lower_range,])

# No Outliers

boxplot(Admission$SOP)

# Only selecting columns which are useful for analysis
Admission<-Admission%>%select(GREScore,TOEFLScore,UniversityRating,SOP,LOR,CGPA,Research,AdmitProb)

# Scatter plot, Frequency curve and Correlation values in one graph
#install.packages("GGally")
library(GGally)
ggpairs(Admission)

# Data Visualization of GRE.Score w.r.t Chance.of.Admit

ggplot(Admission,aes(x=GREScore,y=AdmitProb))+geom_point()+geom_smooth()+ggtitle("Chances of Admit vs GRE Score")

ggplot(Admission,aes(x=GREScore,y=AdmitProb,col=Research))+geom_point()+ggtitle("Chances of Admit vs Gre Score based on Research")

ggplot(Admission,aes(x=GREScore,y=AdmitProb,col=SOP))+geom_point()

ggplot(Admission,aes(x=GREScore,y=AdmitProb,col=UniversityRating))+geom_point()
#Data Prepration
#?createDataPartition  # Data Splitting functions
# p - the percentage of data that goes to training
# list -	logical - should the results be in a list (TRUE) or a matrix with the number of rows equal to floor(p * length(y)) and times columns.

# Splitting of data allows to validate your model and check the accuracy of algorithm.
# If you build the model entirely on your available data, then you will never know how good your model is since you don't have a reference. Train & Test sets serve this issue.

# The steps in analysis are:
# 1. Split the data into train and test randomly based on the outcome variable (you want the distribution of your outcome variable to be as similar as possible between your train and test sets)
# 2. Train your model on train set
# 3. Validate your model prediction result on test set to see the performance
# 4. If bad model, repeat 2-3
# 5. If the model is good, re-train your model on all the data and the make prediction on new data coming in

# createDataPartition function form caret package

trainIndex <- createDataPartition(Admission$AdmitProb, p=0.70, list = FALSE)
#print(trainIndex)
admissionTrain <- Admission[trainIndex,]
admissionTest <- Admission[-trainIndex,]

dim(Admission)

dim(admissionTrain)
View(admissionTrain)

dim(admissionTest)
View(admissionTest)

#Model Formation
# "nnet" package for multinom
?multinom
admTrainReg <- multinom(admissionTrain$AdmitProb ~., family = binomial, data = admissionTrain)
# Fits multinomial log-linear models via neural networks.
# admTrainReg is the model object 
#print(admTrainReg)
summary(admTrainReg) 
# unit changes in sop, will decrease the log of odds ratio by 0.16
# if the individual observation is male, then log of odds ratio decrease by 0.2325
# degree of freedom implies the how many independent random variables you have.
# degrees of freedom = no. of observations – no. of predictors

# The null deviance shows how well the response is predicted by the model with nothing but an intercept.
# The residual deviance shows how well the response is predicted by the model when the predictors are included.


#coefficint of female is missing. And in the same way coefficinet for state boston is also missing.
#Residual Deviance: 

#AIC (Akaike’s Information Criteria), as low as possible.
# When a statistical model is used to represent the process that generated the data, 
# the representation will almost never be exact; so some information will be lost by using the model to represent the process. 
# AIC estimates the relative amount of information lost by a given model: the less information a model loses, 
# the higher the quality of that model.


#predicted values will be the probability however the actual values are just 1 or 0.
#so we define the cutoff and if the observed value is above the cutoff, then it is 1 otherwise 0
#cutoff defines the model. if we reduce the cutoff, then sensitivity will increase.

#model Assessment

#?predict
# Apply the model object on each observation to get he probability of each observation.
predictCalculated <- predict(admTrainReg, admissionTrain, type = "prob")
print(predictCalculated) 
# This is the probability of each observation to get admission in college.

# From above we got the probability, however not the binary result. So we set some threshold value
# and the probability value above the threshold value is 1 else 0.
# Lesser the threshold value, better the sensitivity/accuracy of the model.
# So the threshold value defines the model effecienty. 
predictBinary <- ifelse(predictCalculated > 0.5, 1, 0)
print(predictBinary)

#confusion matrix 
# Build confusion matrix to compare the predicted values with actual values.
confMatrix <- table(predictBinary, admissionTrain$AdmitProb)
print(confMatrix)
# Predicted values are at the horizontal axes and actual on vertical.

#install.packages("e1071")
library(e1071)
confusionMatrix(confMatrix, positive = "1")
# Sensitivity measures the proportion of actual positives that are correctly identified as such 
# Sensitivity : 0.8750 means 87.50% of people are correctly identified who got admission.

# Specificity measures the proportion of actual negatives that are correctly identified as such 
# Specificity : 0.8833 means  88.33% of people are correctly identified who do not got admission.

# 95% CI : (0.8345, 0.9144) : With confidence level(CL) of 95%, confidence interval (CI) is (0.8345, 0.9144).


#looking at the ROC Curve

#prediction() : This function is used to transform the input data into a standardized format.
predictStandardized <- prediction(predictBinary, admissionTrain$AdmitProb)
print(predictStandardized)

perf1 <- performance(predictStandardized, measure = "tpr", x.measure = "fpr")
print(perf1)

perf2 <- performance(predictStandardized, measure = "auc")
print(perf2)


plot(perf1,main = "ROC Curve")
abline(a=0,b=1)

# Area under the curve - auc
# ROC stands for Receiver Operating Characteristic.
# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings. 
# it is a comparison of two operating characteristics (TPR and FPR) as the criterion changes.
# The TPR defines how many correct positive results occur among all positive samples available during the test. 
# The FPR defines how many incorrect positive results occur among all negative samples available during the test.
# What is a good ROC value? A rough guide for classifying the accuracy of a diagnostic test is the traditional academic point system": .90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D)
auc <- perf2@y.values[[1]]
print(auc)
#

#GOF measures
#install.packages("pscl")
library(pscl)
pR2(admTrainReg)

#since the McFadden value is 0.5759350 (0.5 < McFadden < 0.6), it is good model.


# Model Validation
TestPredict <- predict(admTrainReg, admissionTest, type = "prob")
TestPredictBinary <- ifelse(TestPredict > 0.5, 1,0)
TestConfMatrix <- table(TestPredictBinary, admissionTest$AdmitProb)
confusionMatrix(TestConfMatrix, positive = "1")

#Finalfile <- cbind(admissionTest, TestPredict)
Finalfile <- cbind(admissionTest, TestPredictBinary)
View(Finalfile)
write.csv(Finalfile,file = "outfile_logistic_rec.csv")


