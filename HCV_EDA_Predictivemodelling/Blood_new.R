
# STAT 515 Final Project
# Name: Aditya Baxi
#       Jainam Jagani
#       Ivan Francis
# Predictive modeling of hepatitis c virus categories using r : Exploring feature importance and model performance 


# Executing libraries
library(tidyverse)
library(caret)
library(ggplot2)
library(reshape2)
require(nnet)
require(foreign)
library(rpart)
library(psych)
library(glmnet)
library(class)
library(ROCR)
library(pROC)
library(randomForest)
library(DMwR2)
library(randomForestExplainer)

# Showing Basic plots and summaries to understand HCV data

#Reading the dataset.
df = read.csv("C:/stat_new/Final/Blood_Data/hcvdat.csv",header = TRUE)

#Making the performance analytics plot by subtracting category datatypes:
df_plot=df[,-c(1,2,4)]
chart.Correlation(df_plot,histogram=TRUE,pch=19)

pairs.panels(df)

#Distribution of Disease Stages based on Category.
df %>%
  ggplot(aes(x = Category)) +
  geom_bar(fill = "steelblue", color = "black", alpha = 0.8) +
  xlab("Disease Stage") +
  ylab("Total Count") +
  ggtitle("Distribution of Disease Stages") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12),
        axis.text.x = element_text(angle = 45, hjust = 1))

#Distribution of Age of individuals in the dataset.
df %>%
  ggplot(aes(x = Age)) +
  geom_histogram(binwidth = 5, fill = "orange", color = "black", alpha = 0.8) +
  xlab("Age (years)") +
  ylab("Total Patient Count") +
  ggtitle("Distribution of Age") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))

#Gender distribution based on males/females.

df %>%
  ggplot(aes(x = Sex)) +
  geom_bar(fill = "red", color = "black", alpha = 0.8) +
  xlab("Genders") +
  ylab("Total Patient Count") +
  ggtitle("Gender Distribution in the dataset.") +
  theme_classic() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.title = element_text(size = 14),
        axis.text = element_text(size = 12))


#Boxplot for ALP levels
df %>%
  ggplot() +
  geom_boxplot(aes(x = "", y = ALP, fill = "ALP")) +
  guides(fill = guide_legend(title = "Liver Enzyme Levels")) +
  xlab("") +
  ylab("Enzyme Level") +
  ggtitle("Distribution of Liver Enzyme Levels")

#Boxplot for AST levels
df %>%
  ggplot() +
  geom_boxplot(aes(x = "", y = AST, fill = "AST")) +
  guides(fill = guide_legend(title = "Liver Enzyme Levels")) +
  xlab("") +
  ylab("Enzyme Level") +
  ggtitle("Distribution of Liver Enzyme Levels")


#Boxplot for ALT levels
df %>%
  ggplot() +
  geom_boxplot(aes(x = "", y = ALT, fill = "ALT")) +
  guides(fill = guide_legend(title = "Liver Enzyme Levels")) +
  xlab("") +
  ylab("Enzyme Level") +
  ggtitle("Distribution of Liver Enzyme Levels")

#Boxplot for GGT
df %>%
  ggplot() +
  geom_boxplot(aes(x = "", y = GGT, fill = "GGT")) +
  guides(fill = guide_legend(title = "Liver Enzyme Levels")) +
  xlab("") +
  ylab("Enzyme Level") +
  ggtitle("Distribution of Liver Enzyme Levels")

#Scatterplot for the most prominent relation/trend we found was AST vs GGT levels . 
df %>%
  ggplot(aes(x = AST, y = GGT)) +
  geom_point() +
  xlab("Aspartate aminotransferase Level") +
  ylab("Cholesterol Level") +
  ggtitle("AST Levels vs Cholesterol")


# Executing Models

set.seed(123)  # Setting Seed

df$Age <- cut(df$Age, breaks = c(20, 30, 40, 50, 60, 70), labels = c("20-29", "30-39", "40-49", "50-59", "60-69"))

df = df[,-1]

summary(df)  # Summaries of HCV dataset

str(df)
df$Category = as.factor(df$Category)
unique(df$Category)
df$Sex = as.factor(df$Sex)

table(df$Category)

# Multinomial model 1

index = createDataPartition(df$Category, p = 0.8, list = FALSE)
train = df[index, ]
test = df[-index, ]
na.omit(train)

model = multinom(Category~.,data=train)
summary(model)


model1 = predict(model, newdata = test)
model1

cm = table(test$Category, model1)    # Confusion Matrix
print(cm)
accuracy = sum(diag(cm))/sum(cm)     # Accuracy
print(accuracy)


df$hcv = ifelse(df$Category == "0=Blood Donor",0,1)  
table(df$hcv)
df$hcv = as.factor(df$hcv)

pairs.panels(df)


###############################
#Random forest
###############################

df = df[,-1]
index = createDataPartition(df$hcv, p = 0.7, list = FALSE)
train = df[index, ]
test = df[-index, ]
train <- na.omit(train)

rf_model = randomForest(hcv~.,mtry=3, data = train)
summary(rf_model)

#Variable Importance Plot
importance(rf_model)
varImpPlot(rf_model)
########

# Make predictions on the testing set using the random forest model
predicted_quality = predict(rf_model, newdata = test)

cm = table(test$hcv, predicted_quality)
print(cm)
accuracy = sum(diag(cm))/sum(cm)
print(accuracy)

###########
#define object to plot
rocobj= roc(test$hcv, as.numeric(predicted_quality),smoothed = TRUE,plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
            print.auc=TRUE, show.thres=TRUE)


####################
#KNN
####################

df1 = subset(df,select= -Sex)
trial_sum <- rep(0, 20)
trial_n <- rep(0, 20)
df1 <- na.omit(df1)
str(df1)

# Make sure they match with the column names of df1
colnames(df1)

for (i in 1:100) {
  hcv_sample <- sample(1:nrow(df1), size = nrow(df1) * 0.7)
  hcv_train <- na.omit(df1[hcv_sample,])
  hcv_test <- na.omit(df1[-hcv_sample,])
  test_size <- nrow(hcv_test)
  
  for (j in 1:20) {
    hcv_pred = knn(hcv_train[,2:11],hcv_test[,2:11],hcv_train$hcv, k = j)
    
    trial_sum[j] = trial_sum[j] + sum(hcv_pred == hcv_test$hcv)
    trial_n[j] = trial_n[j] + test_size
  }
}

plot(1 - trial_sum / trial_n, type = "l", ylab = "Error Rate", xlab = "K")

which.min(1 - trial_sum / trial_n)

hcv_pred <- knn(train = hcv_train[,2:11], test = hcv_test[,2:11],   # Taking K = 3
                cl = hcv_train$hcv, k = 3)
table(hcv_test$hcv,hcv_pred)

# Convert hcv_test$hcv to factor with levels 0 and 1
hcv_test$hcv <- factor(hcv_test$hcv, levels = c(0, 1))

# Convert hcv_pred to factor with levels 0 and 1
hcv_pred <- factor(hcv_pred, levels = c(0, 1))

# Create confusion matrix
confusionMatrix(hcv_test$hcv, hcv_pred)


#define object to plot
rocobj3 = roc(hcv_test$hcv, as.numeric(hcv_pred),smoothed = TRUE,plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
              print.auc=TRUE, show.thres=TRUE)



##################
# New model for multinomial
##################

model5 = multinom(hcv~.,data=train)
summary(model5)


model6 = predict(model5, newdata = test)
model6


cm2 = table(test$hcv, model6)          # Confusion Matrix
print(cm2)
accuracy2 = sum(diag(cm2))/sum(cm2)   # Accuracy
print(accuracy2)

# Making predictions on the testing set using the multinomial model
predicted_quality3 = predict(model5, newdata = test)

rocobj4 = roc(test$hcv, as.numeric(predicted_quality3),smoothed = TRUE,plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
              print.auc=TRUE, show.thres=TRUE)

##############


