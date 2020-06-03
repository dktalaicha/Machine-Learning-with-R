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
library(e1071)
library(pscl)
library(ggcorrplot)
library(ggplot2) 
library(cowplot)

#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/3_decision_tree")
getwd()

#Importing the csv data
Admission <- read.csv("university_admission_discrete.csv")
#Admission
#View(Admission)
head(Admission)
tail(Admission)

dim(Admission)

Admission<-Admission[,-1]  # remove SerialNo from data set
head(Admission)
#View(Admission)

#Cleaning the data
colnames(Admission)
colnames(Admission) <- str_replace_all(colnames(Admission),"[.]","")
colnames(Admission)


#Dataframe Summary
dfSummary(Admission)

colnames(Admission)

Admission[is.na(Admission$GREScore),1] <- round(mean(Admission$GREScore,na.rm = TRUE))
Admission[is.na(Admission$TOEFLScore),2] <- round(mean(Admission$TOEFLScore, na.rm = TRUE))
Admission[is.na(Admission$UniversityRating),3] <- round(mean(Admission$UniversityRating, na.rm = TRUE))
Admission[is.na(Admission$CGPA),6] <- round(mean(Admission$CGPA, na.rm = TRUE))

dfSummary(Admission)


str(Admission)

#Converting response variable to a factor
class(Admission$AdmitProb)
# Conceptually, factors are variables in R which take on a limited number of different values; 
# such variables are often refered to as categorical variables. One of the most important uses of factors is in statistical modeling; 
# since categorical variables enter into statistical models differently than continuous variables, storing data as factors insures 
# that the modeling functions will treat such data correctly.

#Since the class of dependent variable is integer, so we need to change it to factor
Admission$AdmitProb <- as.factor(Admission$AdmitProb)
class(Admission$AdmitProb)

# checking the splitting is very important. Atleast 60:40 should be there for zero's and one's.
# the predictive model developed using conventional machine learning algorithms could be biased and inaccurate.
table(Admission$AdmitProb)
# The predictive model developed using conventional machine learning algorithms on imbalanced datasets could be biased and inaccurate. 
# If dataset is balanced, then go ahead, otherwise read - https://www.analyticsvidhya.com/blog/2017/03/imbalanced-data-classification/



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

set.seed(123) # to provide the static starting point. 
trainIndex <- createDataPartition(Admission$AdmitProb, p=0.70, list = FALSE)
#print(trainIndex)
admissionTrain <- Admission[trainIndex,]
admissionTest <- Admission[-trainIndex,]

dim(Admission)

dim(admissionTrain)
table(admissionTrain$AdmitProb)
class(admissionTrain$AdmitProb)
#View(admissionTrain)

dim(admissionTest)
table(admissionTest$AdmitProb)
class(admissionTest$AdmitProb)
#checking the splitting is very important. Atleast 60:40 should be there for zero's and one's.
#View(admissionTest)

class(admissionTrain)

#Decision Tree Model
#install.packages("party")
library(party)
tree <- ctree(AdmitProb~., data = admissionTrain, controls = ctree_control(mincriterion = 0.95, minsplit = 10))
print(tree)
plot(tree)
# TrainTree2 <- ctree_control(teststat = c("quadratic", "maximum"), 
#                             testtype = c("Bonferroni", "MonteCarlo", "Univariate", "Teststatistic"), 
#                             mincriterion = 0.95, minsplit = 20, minbucket = 7, 
#                             stump = FALSE, nresample = 9999, maxsurrogate = 0, 
#                             mtry = 0, savesplitstats = TRUE, maxdepth = 0, remove_weights = FALSE)
# plot(TrainTree2)

# mincriterion -  is actually the confidence interval. 75% confidence interval and allowing 25% error.
# minsplit = 25, means that only cut when there are atleast 25 observations in each node.
# if you increase the CI, then tree will shrink.
#last two arguments also called pruning the tree. Increase obs, only then the tree splits.

# mincriterion - the value of the test statistic (for testtype == "Teststatistic"), or 1 - p-value (for other values of testtype) that must be exceeded in order to implement a split.
# 
# minsplit - the minimum sum of weights in a node in order to be considered for splitting.
# 
# minbucket - the minimum sum of weights in a terminal node.

# condition - mincriterion = 0.95, minsplit = 10
# This graph shows that a student who have scored more than 318 in the GRE test and conducted a Research have probability 
# of about 93% of getting admission in University. 

# Also this shows that a student having score more than 311 in GRE and have atleast 3 letter of recommandation (LOR)
# have about 85% chances of getting admission to a universtiy having more than 2 rating.


#Prediction -Train
#predictTrain1 <- predict(tree, admissionTrain, type = "prob")
predictTrain <- predict(tree, admissionTrain)
print(predictTrain) 
head(predictTrain)
head(admissionTrain$AdmitProb)
# This is the probability of each observation to get admission in college on Train dataset.


#Prediction Test
#predictTest1 <- predict(tree, admissionTest, type = "prob")
predictTest <- predict(tree, admissionTest)
print(predictTest) 
# This is the probability of each observation to get admission in college on Test dataset.


# Missclassification error
# Now calculate the Accuracy, Sensitivity, Specificity, Error Rate from Train and Test dataset and compare them.

#Train data
TrainTable <- table(predictTrain, admissionTrain$AdmitProb)
print(TrainTable)
Error <- 1-sum(diag(TrainTable))/sum(TrainTable)
print(Error)
confusionMatrix(TrainTable, positive = "1")
View(admissionTrain)
# Train Data set - Accuracy : 86.83%, Sensitivity - 87.88%, Specificity - 85.34, Error Rate - 13.17%


#Train data
TestTable <- table(predictTest, admissionTest$AdmitProb)
print(TestTable)
Error <- 1-sum(diag(TestTable))/sum(TestTable)
print(Error)
confusionMatrix(TestTable, positive = "1")
View(admissionTest)
# Test Data set - Accuracy : 83.19%, Sensitivity - 81.43%, Specificity - 85.71, Error Rate - 16.81%

# Accuracy has been decreases by about 3.64%,  Sensitivity decreases by 6.45%, Specificity increase by 0.37% and Error Rate increases by 3.64%
# Try to change the mincriterion, minsplit in Decision Tree and check if the difference between Train and Test dataset might be lowered.

#Finalfile 
Finalfile <- cbind(admissionTrain, predictTrain)
View(Finalfile)
write.csv(Finalfile,file = "outfile_decision_tree.csv")


