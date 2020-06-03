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
setwd("/home/dinesh/Data-Science/RStudy/4_random_forest")
getwd()

#Importing the csv data
Admission <- read.csv("university_admission_discrete.csv")
#Admission
#View(Admission)
head(Admission)
tail(Admission)

dim(Admission)

Admission<-Admission[,-1]  # remove SerialNo from data set.
#View(Admission)

#Cleaning the data
colnames(Admission)
colnames(Admission) <- str_replace_all(colnames(Admission),"[.]","")
colnames(Admission)


#Dataframe Summary
dfSummary(Admission)

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


#Random Forest
#rndFrst <- randomForest(AdmitProb~., data = admissionTrain, ntree = 750, mtry = 3, importance = TRUE, prOximity=TRUE, na.action=na.roughfix)
rndFrst <- randomForest(AdmitProb~., data = admissionTrain, ntree = 850, mtry = 3, importance = TRUE)
# mtry is variable randomly selected for each tree, given by square root of features.
# random forest uses majority fote of trees for cliassification of samples, and average for count
# Classification of particular observations based on vote majority by multiple trees.

# ntree	- Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.
# mtry - Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)

?randomForest

summary(rndFrst)
print(rndFrst)
plot(rndFrst)
# Random Forest grap draws the number of decision trees on the horizontal axes and their erros rates on vertical axes.
# By this graph we can predict at which number of trees stablizes the errors rate.
# So randomForest() function helps to decide the optimal number of trees.

# Out-of-bag (OOB) error - OOB is the mean prediction error on each training sample xᵢ, using only the trees that did not have xᵢ in their bootstrap sample.[1]

#Prediction -Train
predictTrain <- predict(rndFrst, admissionTrain)
print(predictTrain) 
head(predictTrain)
head(admissionTrain$AdmitProb)
# This is the probability of each observation to get admission in college on Train dataset.


#Prediction Test
predictTest <- predict(rndFrst, admissionTest)
print(predictTest) 
head(predictTest)
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
# Random Forest Train Data set - Accuracy : 99.29%, Sensitivity - 99.39%, Specificity - 99.14, Error Rate - 0.71%
# Decision Tree Train Data set - Accuracy : 86.83%, Sensitivity - 87.88%, Specificity - 85.34, Error Rate - 13.17%

# As seen above, the accuracy has increased drastically when we have done the analysis using random forest as compared to decision tree.


#Train data
TestTable <- table(predictTest, admissionTest$AdmitProb)
print(TestTable)
Error <- 1-sum(diag(TestTable))/sum(TestTable)
print(Error)
confusionMatrix(TestTable, positive = "1")
View(admissionTest)
# Random Forest Test Data set - Accuracy : 82.35%, Sensitivity - 87.14%, Specificity - 75.51, Error Rate - 17.65%
# Decison Tree Test Data set - Accuracy : 83.19%, Sensitivity - 81.43%, Specificity - 85.71, Error Rate - 16.81%

# Accuracy has been decreases by about 3.64%,  Sensitivity decreases by 6.45%, Specificity increase by 0.37% and Error Rate increases by 3.64%
# Try to change the mincriterion, minsplit in Decision Tree and check if the difference between Train and Test dataset might be lowered.

#importance of variables
varImpPlot(rndFrst, sort = T, n.var = 9)
# This shows the ranking of variables based on the decline in accuracy if we take
# tem out. so the top variable is the most important as it has the largest decline in accuracy.
# The same is also true when uderstanding mean decrease in Gini. The variables used most have largest declines in
#accuracy and gini when removed.

### MeanDecreaseAccuracy
# This graph shows that if we take out GRE Score from the model, then the accuracy of model will decrease by 35%. So GRE Score is very important to get admission in university.
# If we take our the LOR from the model, then the accuracy of the model will decrease by 25%. So LOR plays 2nd important factor in getting admission in university.
# SOP, State, University Rating, Gender are actually do not play significat role for getting admission in university.

### MeanDecreaseGini
# This graph shows that if we take out GRE Score from the model, then the Gini of model will decrease by 35%. So GRE Score is very important to get admission in university.
# If we take our the TOEFL Score from the model, then the accuracy of the model will decrease by 23%. So TOEFL Score plays 2nd important factor in getting admission in university.
# SOP, Research, University Rating, State are actually do not play significat role for getting admission in university.

# From both graphs MeanDecreaseAccuracy and MeanDecreaseGini, we got that GRE Score, Research, LOR, TOEFL Score and CGPA plays important role for getting admission in university.

importance(rndFrst)

#Finalfile 
Finalfile <- cbind(admissionTrain, predictTrain)
View(Finalfile)
write.csv(Finalfile,file = "outfile_random_forest.csv")



