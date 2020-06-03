# https://www.kaggle.com/farazrahman/telco-customer-churn-logisticregression
# https://towardsdatascience.com/data-cleaning-with-r-and-the-tidyverse-detecting-missing-values-ea23c519bc62

## Importing packages
library(tidyverse) 
library(MASS)
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
library(e1071)
library(pscl)
library(ggcorrplot)
library(ggplot2) 
library(cowplot)

#install.packages("MASS")

#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/2_logistic_regression")
getwd()

#Importing the csv data
telecom <- read.csv("Fn-UseC_-Telco-Customer-Churn.csv")
head(telecom)
View(telecom)


str(telecom)
summary(telecom)

# SeniorCitizen is in 'int' form, that can be changed to categorical.
telecom$SeniorCitizen <- as.factor(ifelse(telecom$SeniorCitizen==1, 'YES', 'NO'))


glimpse(telecom)

summary(telecom)

p1 <- ggplot(telecom, aes(gender))+ geom_bar()
p2 <- ggplot(telecom, aes(SeniorCitizen))+ geom_bar()
p3 <- ggplot(telecom, aes(Partner)) + geom_bar()
p4 <- ggplot(telecom, aes(Dependents)) + geom_bar()
p5 <- ggplot(telecom, aes(PhoneService)) + geom_bar()
p6 <- ggplot(telecom, aes(MultipleLines)) + geom_bar()
# simple grid
plot_grid(p1, p2, p3, p4,p5,p6)

p7 <- ggplot(telecom, aes(InternetService)) + geom_bar()
p8 <- ggplot(telecom, aes(OnlineSecurity)) + geom_bar()
p9 <- ggplot(telecom, aes(OnlineBackup)) + geom_bar()
p10 <- ggplot(telecom, aes(DeviceProtection)) + geom_bar()
p11 <- ggplot(telecom, aes(TechSupport)) + geom_bar()
p12 <- ggplot(telecom, aes(StreamingTV)) + geom_bar()
# simple grid
plot_grid(p7,p8,p9,p10,p11,p12)

churnGraph <- ggplot(telecom, aes(Churn)) + geom_bar()
print(churnGraph)


theme1 <- theme_bw()+ theme(axis.text.x = element_text(angle = 0, hjust = 1, vjust = 0.5),legend.position="none")
theme2 <- theme_bw()+ theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),legend.position="none")

theme1 <- theme_bw() + theme(plot.title = element_text(color = "darkred"))

options(repr.plot.width = 6, repr.plot.height = 4)
telecom %>% 
  group_by(Churn) %>% 
  summarise(Count = n())%>% 
  mutate(percent = prop.table(Count)*100)%>%
  ggplot(aes(reorder(Churn, -percent), percent), fill = Churn)+
  geom_col(fill = c("#FC4E07", "#E7B800"))+
  geom_text(aes(label = sprintf("%.2f%%", percent)), hjust = 0.01,vjust = -0.5, size =3)+ 
  theme_bw()+  
  xlab("Churn") + 
  ylab("Percent")+
  ggtitle("Churn Percent")


options(repr.plot.width = 12, repr.plot.height = 8)
plot_grid(ggplot(telecom, aes(x=gender,fill=Churn))+ geom_bar()+ theme1, 
          ggplot(telecom, aes(x=SeniorCitizen,fill=Churn))+ geom_bar(position = 'fill')+theme1,
          ggplot(telecom, aes(x=Partner,fill=Churn))+ geom_bar(position = 'fill')+theme1,
          ggplot(telecom, aes(x=Dependents,fill=Churn))+ geom_bar(position = 'fill')+theme1,
          ggplot(telecom, aes(x=PhoneService,fill=Churn))+ geom_bar(position = 'fill')+theme1,
          ggplot(telecom, aes(x=MultipleLines,fill=Churn))+ geom_bar(position = 'fill')+theme1+
            scale_x_discrete(labels = function(x) str_wrap(x, width = 10)),
          align = "h")

ggplot(telecom, aes(x=gender,fill=Churn))+ geom_bar()+ theme_bw()

# From InternetService graph, Churn rate is much higher in case of Fiber Optic InternetServices.

# There are three continuous variables and they are Tenure, MonthlyCharges and TotalCharges. 


# Checking Normalality 
# Plot graph only for numeric values, not for binomacial or categorical values.

hist(telecom$tenure, col = "orange")
hist(telecom$MonthlyCharges, col = "blue")
hist(telecom$TotalCharges, col = "grey")
# Histogram for above three variables are not bell curve, so not normally distributed.

#Do shapiro test with only the first 5000 records, because for shapiro test sample size must be between 3 and 5000.
#Checking Normanity by shapiro 
shapiro.test(telecom$tenure[0:5000])
# Null Hypothesis : Tensure is normally distributed. Alt Hypotheses : Tensure is not normally distributed.
# Reject null hypothesis since p-value < 0.05. So Tenure is not normally distributed.

shapiro.test(telecom$MonthlyCharges[0:5000])
shapiro.test(telecom$TotalCharges[0:5000])
# Same way, Monthly Charges & Total Charges are also not normally distributed.

#Checking Outliears - BoxPlot – Check for outliers
boxplot(telecom$tenure)  #no outlier for tenure
boxplot(telecom$MonthlyCharges) #no outlier for MonthlyCharges
boxplot(telecom$TotalCharges) #no outlier for TotalCharges


#Dataframe Summary
dfSummary(telecom)

colnames(telecom)

#Replace with missing values of columns with their means
telecom[is.na(telecom$TotalCharges),19] <- round(mean(telecom$TotalCharges,na.rm = TRUE))


dfSummary(telecom)

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

# Splitting the data set into 70:30 ratio for Train:Test
trainIndex <- createDataPartition(telecom$Churn, p=0.70, list = FALSE)
print(trainIndex)
# 70% Train dataset for regression analysis
telecomTrain <- telecom[trainIndex,]
# Remaining 30% Test dataset for testing
telecomTest <- telecom[-trainIndex,]

dim(telecom)

dim(telecomTrain)
table(telecomTrain$Churn)
View(telecomTrain)

dim(telecomTest)
table(telecomTest$Churn)
View(telecomTest)

#Model Formation
# "nnet" package for multinom
# Multinomial regression function : This algorithm allows to predict a categorical dependent variable which has more than two levels. 
# Training the multinomial model
telecomTrainReg <- multinom(telecomTrain$Churn ~., family = binomial, data = telecomTrain)

# telecomTrainReg is the model object 
#print(telecomTrainReg)

# summary() function to explore the beta coefficients of the model.
summary(telecomTrainReg) 
# Important : The output coefficients are represented in the log of odds.
# Each row in the coefficient table corresponds to the model equation.
# unit changes in tenure, will decrease the log of odds ratio by 0.0578153259
# if the individual observation is male, then log of odds ratio decrease by 0.0019358705
# degree of freedom implies the how many independent random variables you have.
# degrees of freedom = no. of observations – no. of predictors

# The null deviance shows how well the response is predicted by the model with nothing but an intercept.
# The residual deviance shows how well the response is predicted by the model when the predictors are included.


#coefficint of female is missing. 
#Residual Deviance: 
#AIC (Akaike’s Information Criteria), as low as possible


#predicted values will be the probability however the actual values are just 1 or 0.
#so we define the cutoff and if the observed value is above the cutoff, then it is 1 otherwise 0
#cutoff defines the model. if we reduce the cutoff, then sensitivity will increase.

#model Assessment

#?predict
# Apply the model object on each observation to get he probability of each observation.
predictCalculated <- predict(telecomTrainReg, telecomTrain, type = "prob")
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
confMatrix <- table(predictBinary, telecomTrain$Churn)
print(confMatrix)
# Predicted values are at the horizontal axes and actual on vertical.

#install.packages("e1071")

confusionMatrix(confMatrix, positive = "1")
# Accuracy : 0.8065 means 80.65% predicted value is accurate from actual value in training dataset.   
# Sensitivity measures the proportion of actual positives that are correctly identified as such 
# Sensitivity : 0.5615 means 56.15% of subscribers are correctly identified who moved/churned from a specific service or a service provider.

# Specificity measures the proportion of actual negatives that are correctly identified as such 
# Specificity : 0.8966 means  89.66% of subscribers are correctly identified who do not moved/churned to other service provider.

# 95% CI : (0.7952, 0.8175) : With confidence level(CL) of 95%, confidence interval (CI) is (0.7952, 0.8175).

# As we can see above, when we are using a cutoff of 0.50, we are getting a good accuracy and specificity, but the sensitivity is very less. 

#looking at the ROC Curve

#prediction() : This function is used to transform the input data into a standardized format.
predictStandardized <- prediction(predictBinary, telecomTrain$Churn)
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
# Since the Area under the curve is only 72.90%, so it is not very good model.

#GOF measures
#install.packages("pscl")

pR2(telecomTrainReg)

#since the McFadden value is 0.2906506 (0.5 < McFadden < 0.6), it is very bad model.


# Model Validation
TestPredict <- predict(telecomTrainReg, telecomTest, type = "prob")
TestPredictBinary <- ifelse(TestPredict > 0.5, 1,0)
TestConfMatrix <- table(TestPredictBinary, telecomTest$Churn)
confusionMatrix(TestConfMatrix, positive = "1")
# Accuracy : 0.8007, Accuracy in testing dataset is 80.07% which is almost same for traing dataset 80.65%.  
# Sensitivity : 0.5368 means 53.68% of subscribers are correctly identified who moved/churned from a specific service or a service provider.
# Specificity : 0.8922 means  89.22% of subscribers are correctly identified who do not moved/churned to other service provider.
# 95% CI : (0.783, 0.8175): With confidence level(CL) of 95%, confidence interval (CI) is (0.783, 0.8175).

# Accuracy, Sensitivity, Specificity and CI are allmost same for test dataset as compared to train dataset.


#Finalfile <- cbind(telecomTest, TestPredict)
Finalfile <- cbind(telecomTest, TestPredictBinary)
View(Finalfile)
write.csv(Finalfile,file = "outfile_logistic_telecom.csv")


