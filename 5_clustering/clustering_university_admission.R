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
setwd("/home/dinesh/Data-Science/RStudy/5_clustering")
getwd()

#Importing the csv data
Admission <- read.csv("university_admission_discrete.csv")
#Admission
View(Admission)
head(Admission)
tail(Admission)

dim(Admission)

#Cleaning the data
colnames(Admission)
colnames(Admission) <- str_replace_all(colnames(Admission),"[.]","")
colnames(Admission)


#Dataframe Summary
dfSummary(Admission)

Admission[is.na(Admission$GREScore),2] <- round(mean(Admission$GREScore,na.rm = TRUE))
Admission[is.na(Admission$TOEFLScore),3] <- round(mean(Admission$TOEFLScore, na.rm = TRUE))
Admission[is.na(Admission$UniversityRating),4] <- round(mean(Admission$UniversityRating, na.rm = TRUE))
Admission[is.na(Admission$CGPA),7] <- round(mean(Admission$CGPA, na.rm = TRUE))

dfSummary(Admission)


str(Admission)

#Heirarchical clustering

# Normalization
colnames(Admission)
head(Admission)
admissionContinous <- Admission[,-c(1,8,9,10,11)]
# Removing qualatative variables because cluster works best when variables are quantative/numeric in nature.
# Because calculating distance between qualatative variables are useless they have value for examaple 0 or 1, and distance between 0 & 1 will always be same.
# Keep only continous variables.

View(admissionContinous)
# In the above dataset, the values of UniversityRating, SOP, LOR and CGPA is not varying much. Their values varies between 1 to 10.
# However the GREScore and TOEFLScore varies in large range. 

# So some variables are varing in wider range while some are just varing in smaller range. So we need to bring them at a common comparing level.

# Brining the  observation to common level with reference to mean and std - Scalling
Mean <- lapply(admissionContinous, mean)
print(Mean)

SD <- lapply(admissionContinous, sd)
print(SD)


admissionScale <- scale(admissionContinous, Mean, SD)
View(admissionScale)
# some observations are positive while some are negative. 
# Positive means they are more then their mean while negative means they are less then their mean value.
# During scaling, every value is calculated from a reference value mean.


#Euclidean distance 
# calculate distance between datapoints on scalled dataset only.
distance <- dist(admissionScale)
print(distance)

#--------------------- Hierarchical Clustering Technique ---------------------#
# Cluster dendrogram
# A dendrogram is a diagram that shows the hierarchical relationship between objects.
hcc <- hclust(distance)
plot(hcc)
# The clustering has grouped all the observations into certain distinct branches based on the similar features. 
# There should be homogeneity within the cluster however heterogeneity among the clusters.
# The number in the graph shows which observation is allocated to which cluster. 
# Usually we cut the graph at the height of 4 to get more granularity.

#cluster membership
memberc <- cutree(hcc, 3)
print(memberc)
# memberc shows the observations goes to which cluster. 
# So observation 1st, 2nd goest to first cluster; observation 2nd, 3rd goes to 2nd cluster. Observation 9 goest to 3rd cluster.
# Students belongs to same cluster have similar characteristics. 
table <- cbind(admissionContinous, memberc)
View(table)


# cluster means
aggregate(admissionScale, list(memberc), mean)
# negative values means lower than average, positive is above average, consider by variance.
# Students belongs to cluster 1 are high performing and have more chances of getting admission in university.
# Students belongs to cluster 3 are poor performing and have low changes of getting admission in university.
aggregate(admissionContinous, list(memberc), mean)
# This the obervations based on the original data based on their means.
# so Student having average GRE score of 331 or more, average TOEFL score of 114 or more, have atleat average LOR of about 4, and have average CGPA score of 9.5 or more have high changes of getting admission in a university having average rating 4.5 and above.

#silhouette plot
library(cluster)
plot(silhouette(cutree(hcc,3),distance))
# hight Si values mean that the cluster are good, also negative values mean the value is 
# out of place in the cluster.

#--------------------- K-Means Clustering Technique ---------------------#

KcScale <- kmeans(admissionScale,3)
print(KcScale)
#Gives the members of the cluster, the number of memebers and within cluster sum of squares.
# This should be a slow as possible indicating a homogenuous cluster.
# Between sum of squares should be as large as possible.

KcScale$cluster  
# shows allocation of each observation to a particular cluster
KcScale$centers  
# shows values of each variables of the cluster. So cluster 3 is high performing while cluster 1 is worst performing.
KcScale$withinss 
# Shows the total variance within each clusters. Each Clusters should be homogenues, so withinss should be as low as possible. 
# So cluster 2 having variance of 12.64558 is more homogenues.
KcScale$tot.withinss
# Sum of above variance. Shows the total of variance within the cluster. This should be low.
KcScale$betweenss
# Shows  the total of variance among the cluster. All cluster should be heterognues, so this should be high.
KcScale$totss    
# total sum of squares -  Shows the total variance within and among the cluster.
# totss = tot.withinss + betweenss
KcScale$size
# No of observations in each cluster.
KcScale$iter
KcScale$ifault

KcContinues <- kmeans(admissionContinous,3)
print(KcContinues)

KcContinues$cluster
KcContinues$centers
KcContinues$withinss
KcContinues$tot.withinss
KcContinues$totss
KcContinues$size
KcContinues$iter
KcContinues$ifault

plot(admissionContinous$GREScore, admissionContinous$CGPA, col = KcScale$cluster)
# this shows the scattering of each observations. Black color dots shows the observations belong to one cluster, green ones to another cluster and red belongs to another cluster.
# Black dots are more or less are together, and same for others.
plot(admissionContinous$GREScore, admissionContinous$TOEFLScore, col = KcScale$cluster)

