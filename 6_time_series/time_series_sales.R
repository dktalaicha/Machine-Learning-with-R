#install.packages("forecast")
library(forecast)

#Setting the working directory path 
getwd()
setwd("/home/dinesh/Data-Science/RStudy/6_time_series")

#Importing the csv data
data <- read.csv("Sales.csv")
data
head(data)
View(data)

class(data)

colnames(data)[c(1:2)] <- c("date","sales")

## data preparation steps
# converting into timeseries data
data <- ts(data[,2],start = c(2003,1),frequency = 12)
class(data)
print(data)

#plotting the data
plot(data, xlab = "Years", ylab = "Sales")

#Differencing the data to get rid off the trend (correcting for non stationarity)
plot(diff(data), ylab = "Differenced Sales")

##Testing for Stationarity
#Augmented Dickey-Fuller (ADF) Test
library(tseries)
adf.test(diff(data), alternative = "stationary")

#Automated ARIMA modelling
require(forecast)
ARIMAfit1 <- auto.arima(diff(data), approximation = TRUE, trace = TRUE)
print(ARIMAfit1)
summary(ARIMAfit1)
# Lower the AIC and BIC the better the model

# ACF and PACF plots
acf(ARIMAfit1$residuals, main = "correlogram")
#Shows autocorrelation for in sample forecast errors should not exceed bands the dotted lines.
pacf(ARIMAfit1$residuals, main = "partial correlogram")
# if lines exceed the bands then there is autocorrelation possibility and we use the Ljung Box test.

#Ljung Box test
Box.test(ARIMAfit1$residuals, lag = 20, type = "Ljung-Box")
#Null Hypothesis of no autocorrelation.
# p-values > 0.5, accept null hypothesis. So there is no AutoCorrelation.

#Residuals Plot
ARIMAfit1$residuals
hist(ARIMAfit1$residuals, col = "red", xlab = "Error", main = "Histogram", freq = FALSE)
lines(density(ARIMAfit1$residuals))

#Forest
library(ggplot2)
f <- forecast(ARIMAfit1,36)
print(f)
autoplot(f)
accuracy(f)
