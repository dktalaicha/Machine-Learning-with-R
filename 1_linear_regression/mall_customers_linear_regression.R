#Load required library or packages

library(zoo) 
library(carData) 
library(car) 
library(lmtest) 
library(nortest)
library(dplyr) 
library(stringr)
library(summarytools)
library(ggplot2)

#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/1_linear_regression")

#Importing the csv data
mall <- read.csv("mall_customers.csv")
mall
str(mall)
View(mall)

# Checking the dimensions(no of rows and no of columns) of data.
dim(mall)

#Cleaning the data
colnames(mall)
colnames(mall) <- str_replace_all(colnames(mall),"[.]","") #replacing . with blank
colnames(mall)

#Checking Normalality for annual income and age.
hist(mall$AnnualIncomek,col = "green")
hist(mall$Age,col = "orange")
hist(mall$Gender, col = "red")  #Histogram cannot be ploted again Nominal Variables.

#Checking Normanity by shapiro 
#?shapiro.test
#The Shapiro-Wilks test for normality. 
#The Shapiro–Wilk test tests the null hypothesis that a sample x1, ..., xn came from a normally distributed population.
#This test rejects the hypothesis of normality when the p-value is less than or equal to chosen alpha level(0.05).
#The null-hypothesis of this test is that the population is normally distributed.
#Thus, if the p value is less than the chosen alpha level, then the null hypothesis is rejected and 
#there is evidence that the data tested are NOT normally distributed. 
#On the other hand, if the p value is greater than the chosen alpha level(0.05), then the null hypothesis that the data came from a normally distributed population can not be rejected 

shapiro.test(mall$AnnualIncomek)  # Null Hypothesis - Annual income is normally distributed.
shapiro.test(mall$Age) # Null Hypothesis - Age is normally distributed.
shapiro.test(mall$Gender)  #shapiro.test or Normality test cannot be calculated for Nominal or Discreate Variables.  Normality test can be done only for continues values.

########### summary of shapiro test ################

# p-value of annual income and age is less than 0.05, null hypothesis get rejected.
# so both anuuual income and age, there are evidence that the both variables tested are NOT normally distributed.

####################################################


#Dataframe Summary
#Summary of a dataframe consisting of: variable names and labels, factor levels, frequencies or numerical summary statistics, 
#and valid/missing observations information.

#dfSum <- dfSummary(mall)
#View(dfSum)
dfSummary(mall)


colnames(mall)

#Replace with missing values of columns with their means
mall[is.na(mall$Age),3] <- round(mean(mall$Age,na.rm = TRUE))
mall[is.na(mall$AnnualIncomek),4] <- round(mean(mall$AnnualIncomek, na.rm = TRUE))
mall[is.na(mall$SpendingScore1100),5] <- round(mean(mall$SpendingScore1100, na.rm = TRUE))
View(mall)

#Cross check again if missing values are replaced with mean or not
dfSummary(mall)


#Checking Outliears - BoxPlot – Check for outliers
boxplot(mall$Age)  #no outlier for age
boxplot(mall$AnnualIncomek) # there is a outlier for annual income

#Get the details of outliear in annual income
boxplot.stats(mall$AnnualIncomek)


# scatterplot of the variables
scatter.smooth(x=mall$AnnualIncomek, y=mall$SpendingScore1100,xlab = "Annual Income", ylab = "Spending Score",main="Annual Income vs Spending Score", col='blue', pch=20, cex=2)
scatter.smooth(x=mall$Gender, y=mall$SpendingScore1100,xlab = "Gender", ylab = "Spending Score",main="Gender vs Spending Score", col='blue', pch=20, cex=2)
scatter.smooth(x=mall$Age, y=mall$SpendingScore1100, xlab = "Age", ylab = "Spending Score", main="Age vs Spending Score", col='blue', pch=20, cex=2)

#Regression
#lm is used to fit linear models. It can be used to carry out regression, single stratum analysis of variance and analysis of covariance.
#lm(formula, data, subset, .....) , formula: describes the model. A typical model has the form "response ~ terms"
#mall[,-1] <== removing first colurmn it is CustomerId and it doesnot have any impact on dependent variable.
# The lm functions returns coefficients, effects, fitted.values and residuals.

# -1 in below code is to drop the first column "CustomerId", as this doest not have any impact on spending score.

reg1 <- lm(mall$SpendingScore1100 ~., data = mall[,-1]) 
print(reg1)

# Regression Equation: 
#       Spending Score = 72.60 + (-1.918 * Gender) + (-0.564 * Age) + (0.0075 * Annual-Income)
#       SS = 72.60 - 1.918*G - 0.564*A + 0.0075*I

########################
#1. Intercept tells that the spending score will be 72.60 when neither of gender, age and annual income have any impact.
#2. Coefficient of Gender - if the individual observation is male, then spending score decrease by 1.918
#3. Coefficient of Age - 0.56 means that a unit change in age will decreases the spending score by 0.56.
#4. Coefficient of Annual income of 0.0075 means that a unit change in annual income will increate spending score by 0.0075.
#########################


#In case of caterorical variable (here gender), the no of binary columns created by R will be equl to one less than accedity. This is called "Dummy Variable Trap". 
# We got coefficient only for GenderMale instead for both Male & Female. So coefficient for female is lost.

summary(reg1) 

#########################
#  Meaning Behind Each Section of Summary()
# 1. Residuals: Difference between what the model predicted and the actual value of y.  You can calculate the Residuals section like so: summary(y-model$fitted.values)
# 2. Coefficients: These are the weights that minimize the sum of the square of the errors.  
#       2.1 : Estimate : Coefficient of each independent variables along with intercept
#       2.2 : Std. Error is Residual Standard Error divided by the square root of the sum of the square of that particular x variable. 
#       2.3 : t value: Estimate divided by Std. Error
#       2.4 : Pr(>|t|): Look up your t value in a T distribution table with the given degrees of freedom.
# 3. Residual standard error : The Residual Standard Error is the average amount that the response (dist) will deviate from the true regression line. The smaller the standard error, the less the spread and the more likely it is that any sample mean is close to the population mean. A small standard error is thus a Good Thing. 
# 4. Multiple R-squared, Adjusted R-squared : R2  is a measure of the linear relationship between predictor variable and our response variable. 
# 3 The F-Statistic is a “global” test that checks if at least one of your coefficients are nonzero.

# A side note: In multiple regression settings, the R2 will always increase as more variables are included in the model. That’s why the adjusted R2 is the preferred measure as it adjusts for the number of variables considered.	

#******************* Overall Analysis Starts for lm function *******************#
# Null Hypothesis - There is no relationship of spending scores with age, gender, & annual income. 
#                   Or None of Age, Gender and Annual Income are significant for spending score.
#     Ho:Age, Gender, Annual Income = 0


# Alternate Hypothesis - Atlest one of the depended variables (age, gender, income) are significant and can effect the spending score.
#     H1:Age, Gender, Annual Income != 0

# p-value (0.0001796) < 0.05, so rejecting the null hypothesis. So Atlest one of the depended variables (age, gender, income) are significant and can effect the spending score.
# However R2 is very low, only 9.7% of the varinance found in the spending score can be explained by Age, Gender and Income.

# IMPORTANT -  R2/Adj R2 are vey low so we can reject the model, however due to the low value of p-value, null hypotheses get rejected and atleast one independed variable have impact on depended variable. So we cannot say that the model is completely useless however it is not the good model.

# 1. Residual standard error: 24.58 - The actual spending score deviate from the true regression line by approximately 24.58, on average. The smaller the standard error, the less the spread and the more likely it is that any sample mean is close to the population mean. A small standard error is thus a Good Thing.
# 2. Multiple R-squared:  0.0967 - R2 is very less in our case, so it is not a good model. Roughly 9.7% of the varinance found in the spending score can be explained by Age, Gender and Income.
# 3. Adjusted R-squared:  0.0828  - Adj R2 (8.3 %) is also very less in our case, so it is not good model. Actually only 8.3% of the varinance found in the spending score can be explained by Age, Gender and Income.
# Gap between R-squared and Adjusted R-squared is 1.4% only, which is good. Typically the more non-significant variables you add into the model, the gap between two increases.

# F-statistic: 6.958 - The lower the F-statistic, the closer to a non-significant model. So F-statistic is low means it is not very significant model.

#******************* Individuals Variables Analysis *******************#

# Null Hypothesis for Gender - The coefficient of Gender is zero. (or Gender doesnot impact spending score. or Gender is insignificant.)
# Alternate Hypothesis for Gender - The coefficient of Gender is not zero. (or Gender does impact spending score. or Gender is significant.)
# p-value of Gender (0.587) > 0.05 - Accept the null hypothesis, and it means that Gender does Not have statistically significiant impact on Spending. 

# Null Hypothesis for age - The coefficient of age is zero. (or age doesnot impact spending score. or age is insignificant.)
# Alternate Hypothesis for age - The coefficient of age is not zero. (or age does impact spending score. or age is significant.)
# p-value of age < 0.05 - Reject the null hypothesis, and it means that age does have statistically significiant impact on Spending. 


# Null Hypothesis for Annual Income - The coefficient of Annual Income is zero. (or Annual Income doesnot impact spending score. or Annual Income is insignificant.)
# Alternate Hypothesis for Annual Income - The coefficient of Annual Income is not zero. (or Annual Income does impact spending score. or Annual Income is significant.)
# p-value of Annual Income (0.911) > 0.05 - Accept the null hypothesis, and it means that Annual Income does Not have statistically significiant impact on Spending. 

#Gender is insignificant.
#age is significant
#Annual Income is insignificant

# Three Starts *** infront of independed variable show the it is signficant.

#*************************************#

######--------- Residuals Analysis - Null hypothesis: data is normal ------###################

# fitten function calculates the predicated value for all observations based on the above calculated regression equation.
# so it will put the independent variable values in this equation "SS = 72.60 - 1.918*G - 0.564*A + 0.0075*I" and give output for all observations. 
fitted(reg1)

# Now we need to check what the error/residual.
# Residual = Calculated value - Actual value
residual <- reg1$residuals
print(residual)


#Check the normality of Error/Residual Term (Linear Regression assumes that error are normally distributed.)
#Null Hypotheses - Errors are normally distributed. Alt Hypothese - Errors are not normally distributed.
shapiro.test(residual)
# p-value(0.00837) < 0.05, Null Hypotheses get rejected, and so the errors are not normally distributed.
hist(residual,col = "green")

#checking for multicolinearity (checking correlation between independent variables.)
# In our model, only those independent variabel should exist which are not correlated with each other.
# This is done using Correlation Matrix

cor(mall) # giving error due to gender is not numeric

colnames(mall)
mallNumeric<-mall[,-c(1,2)]  # remove CustomerID and Gender from data set.
colnames(mallNumeric)
View(mallNumeric)

cor(mallNumeric)  #calculate the multicolinearity between independed variables.

# if there is high correlation between two independed variables (high multicollinearity), then you will not be able to seperate out the impact of individual independed variable on depended variable.
# Due to multicolinearity we can't define the complete impact of only one independed variable on the depended variable.


#checking for heteroscedasticity (checking for error variance )
# null hypothesis - Errors are homoscedasticity and alternate hypothesis errors are heteroscedasticity
bptest(reg1)
# p-value < 0.05, so it rejects that errors are homoscedasticity. 
# So errors terms are heteroscedasticity and does not have contant variance which is not good for model.

#checking autocorrelation (Checking correlation between errors)
dwt(reg1)
# if D-W Statistic is around 2, then we have autocorrelation in model. and awavy from 2 means no autocorrelation.
# here D-W Statistic is 3.4, so there is no autocorrelation in the model.

#checking for MAPE
# Comparision between actual and predictied results.
predictmall <- predict(reg1,mall)
mallNew <- cbind(mall,predictmall)
View(mallNew)

colnames(mallNew)

#Calculating error rate
Error <- abs((mallNew$SpendingScore1100 - mallNew$predictmall)/(mallNew$SpendingScore1100)*100)
print(Error)

View(mallNew)
mallNew <- cbind(mall,predictmall,Error)
View(mallNew)
#calculating mean of error rate
mean(mallNew$Error, na.rm = TRUE)

# Average error rate of model is 148%, which is very high and we can say that model is worst model.

hist(Error)
boxplot(Error)


# https://rpubs.com/iabrady/residual-analysis

ggplot(mall, aes(x = AnnualIncomek, y = SpendingScore1100)) +
  geom_smooth(method = "lm", se = FALSE, color = "lightgrey") +     # regression line  
  geom_segment(aes(xend = AnnualIncomek, yend = predictmall), alpha = .2) +      # draw line from point to line
  geom_point(aes(color = abs(residual), size = abs(residual))) +  # size of the points
  scale_color_continuous(low = "green", high = "red") +             # colour of the points mapped to residual size - green smaller, red larger
  guides(color = FALSE, size = FALSE) +                             # Size legend removed
  geom_point(aes(y = predictmall), shape = 1) +
  theme_bw()



