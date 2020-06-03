# https://www.kaggle.com/ravichaubey1506/postgraduate-admission-analysis

#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/1_linear_regression")

#Importing the csv data
Admission <- read.csv("university_admission_continous.csv",header=TRUE,sep=",")
head(Admission)
tail(Admission)
# Admission
# View(Admission)

dim(Admission)



#Checking Normalality for TOEFL.Score
hist(Admission$TOEFL.Score,col = "green")

#?shapiro.test
shapiro.test(Admission$TOEFL.Score)

shapiro.test(data$GRE.Score)

#Installing Required Packages
#install.packages("car")
#install.packages("lmtest")
#install.packages("nortest")
#install.packages("dplry")
#install.packages("stringr")


#library(car)
library(lmtest)
library(nortest)
library(dplyr)
library(stringr)


#Cleaning the data
colnames(Admission)
colnames(Admission) <- str_replace_all(colnames(Admission),"[.]","")
colnames(Admission)

#install.packages("summarytools")
library(summarytools)
dfSummary(Admission)

Admission[is.na(Admission$GREScore),2] <- round(mean(Admission$GREScore,na.rm = TRUE))
Admission[is.na(Admission$TOEFLScore),3] <- round(mean(Admission$TOEFLScore, na.rm = TRUE))
Admission[is.na(Admission$UniversityRating),4] <- round(mean(Admission$UniversityRating, na.rm = TRUE))
Admission[is.na(Admission$CGPA),7] <- round(mean(Admission$CGPA, na.rm = TRUE))

dfSummary(Admission)

#Removing Outliears - BoxPlot – Check for outliers
boxplot(Admission$GREScore)
boxplot(Admission$TOEFLScore)
boxplot(Admission$UniversityRating)
boxplot(Admission$SOP)
boxplot(Admission$LOR)
boxplot(Admission$CGPA)

#Regression
reg1 <- lm(Admission$ChanceofAdmit~., data = Admission[,-1]) 
summary(reg1) 
print(reg1)

# Scatter Plot
scatter.smooth(x=Admission$GREScore, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")
scatter.smooth(x=Admission$TOEFLScore, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")
scatter.smooth(x=Admission$UniversityRating, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")
scatter.smooth(x=Admission$SOP, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")
scatter.smooth(x=Admission$LOR, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")
scatter.smooth(x=Admission$CGPA, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")
scatter.smooth(x=Admission$Research, y=Admission$ChanceofAdmit,main="GREScore vs ChanceofAdmit")

#Density plot – Check if the response variable is close to normality
plot(density(Admission$GREScore), main="Density Plot: GREScore", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(Admission$GREScore), 2)))
polygon(density(Admission$GREScore), col="red")

# calculate correlation between GREScore and ChanceofAdmit 
cor(Admission$GREScore, Admission$ChanceofAdmit)  

cor(Admission$TOEFLScore, Admission$ChanceofAdmit) 


