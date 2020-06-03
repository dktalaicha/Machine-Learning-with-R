#Importing packages
library(zoo) 
library(carData) 
library(car) 
library(lmtest) 
library(nortest)
library(dplyr) 
library(stringr)
library(summarytools)
library(caret)


#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/3_decision_tree")
getwd()

#Importing the csv data
telecom <- read.csv("Fn-UseC_-Telco-Customer-Churn.csv")
#View(telecom)


str(telecom)

#Cleaning the data
colnames(telecom)
colnames(telecom) <- str_replace_all(colnames(telecom),"[.]","")
colnames(telecom)


summary(telecom)

#Replace with missing values of columns with their means
telecom[is.na(telecom$TotalCharges),19] <- round(mean(telecom$TotalCharges,na.rm = TRUE))


dfSummary(telecom)

#Converting response variable to a factor
class(telecom$Churn)

#Since the class of dependent variable is integer, so we need to change it to factor
telecom$Churn <- as.factor(telecom$Churn)
class(telecom$Churn)

table(telecom$Churn)
#checking the splitting is very important. Atleast 60:40 should be there.
colnames(telecom)

# Splitting the data set into 70:30 ratio for Train:Test
set.seed(123) # to provide the static starting point.

# Required package for createDataPartition is caret
trainIndex <- createDataPartition(telecom$Churn, p=0.70, list = FALSE)
telecomTrain <- telecom[trainIndex,]
telecomTest <- telecom[-trainIndex,]

class(telecomTrain)


dim(telecomTrain)
table(telecom$Churn)
table(telecomTrain$Churn)
table(telecomTest$Churn)

str(telecomTrain)

library(party)
tree <- ctree(Churn~., data = telecomTrain, controls = ctree_control( mincriterion = 0.95, minsplit = 500))
plot(tree)


#Prediction -Train
predictTrain1 <- predict(tree, telecomTrain, type = "prob")
predictTrain2 <- predict(tree, telecomTrain)

#Missclassification error
#Train data
Traintab <- table(predictTrain2, telecomTrain$Churn)
print(Traintab)
Error <- 1-sum(diag(Traintab))/sum(Traintab)
confusionMatrix(Traintab, positive = "1")
View(telecomTrain)
