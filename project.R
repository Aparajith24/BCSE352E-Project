# CREDIT CARD ATTRITION RATE

# Load necessary libraries
library(tidyverse)
library(ggplot2)

# Read the data from a CSV file
data <- read.csv("/Users/aparajith/Downloads/credit_card_churn.csv")

# Remove the last 2 columns from the dataset
data <- data[, -c(21, 22)]

# Check the dimensions of the dataset
dim(data)

# Display the column names
colnames(data)

# Display the first few rows of the dataset
head(data)

# Display the last few rows of the dataset
tail(data)

# Display the data types of each column
str(data)

# UNIVARIATE ANALYSIS

# Summary statistics of the dataset
summary(data)

# Histogram of Customer Age
hist(data$Customer_Age)

# Bar plot of Attrition Flag
barplot(table(data$Attrition_Flag))

# Bar plot of Education Level
barplot(table(data$Education_Level))

# Bar plot of Income Category
barplot(table(data$Income_Category))

# Bar plot of Card Category
barplot(table(data$Card_Category))

# BIVARIATE ANALYSIS

# Correlation between Customer Age and Credit Limit
cor(data$Customer_Age, data$Credit_Limit)

# Scatter plot of Customer Age vs. Credit Limit
plot(data$Customer_Age, data$Credit_Limit)

# Correlation between Customer Age and Total Transaction Amount
cor(data$Customer_Age, data$Total_Trans_Amt)

# Scatter plot of Customer Age vs. Total Transaction Amount
plot(data$Customer_Age, data$Total_Trans_Amt)

# Aggregate mean Customer Age by Attrition Flag
aggregate(data$Customer_Age, by = list(data$Attrition_Flag), mean)

# Aggregate mean Customer Age by Gender
aggregate(data$Customer_Age, by = list(data$Gender), mean)

# Aggregate mean Total Relationship Count by Card Category
aggregate(data$Total_Relationship_Count, by = list(data$Card_Category), mean)

# Aggregate mean Credit Limit by Gender
aggregate(data$Credit_Limit, by = list(data$Gender), mean)

# Aggregate mean Credit Limit by Income Category
aggregate(data$Credit_Limit, by = list(data$Income_Category), mean)

# Aggregate mean Credit Limit by Card Category
aggregate(data$Credit_Limit, by = list(data$Card_Category), mean)

# Cross tabulation for Gender and Attrition Flag
cross_tab <- table(data$Gender, data$Attrition_Flag)
prop_cross_tab <- prop.table(cross_tab, margin = 1)
barplot(prop_cross_tab, legend = TRUE)

# Cross tabulation for Education Level and Attrition Flag
cross_tab2 <- table(data$Education_Level, data$Attrition_Flag)
prop_cross_tab2 <- prop.table(cross_tab2, margin = 1)
barplot(prop_cross_tab2, legend = TRUE)

# Missing Values & Outlier Treatment

# Check for missing values
sum(is.na(data))

# Boxplot of Customer Age
boxplot(data$Customer_Age)

# Replace outliers in Customer Age with the mean
data$Customer_Age[data$Customer_Age > 68] <- mean(data$Customer_Age)

# Boxplot of Customer Age after outlier treatment
boxplot(data$Customer_Age)

# Replace "Gold" and "Platinum" Card Category with "Silver"
table(data$Card_Category)
data$Card_Category[data$Card_Category == "Gold" | data$Card_Category == "Platinum"] <- "Silver"
table(data$Card_Category)

# Label Encoding

# Convert categorical variables to numeric using label encoding
data$Gender_n <- as.numeric(factor(data$Gender))
data$Education_Level_n <- as.numeric(factor(data$Education_Level))
data$Marital_Status_n <- as.numeric(factor(data$Marital_Status))
data$Income_Category_n <- as.numeric(factor(data$Income_Category))
data$Card_Category_n <- as.numeric(factor(data$Card_Category))

# Remove original categorical variables
data <- data[, -c(17:21)]

# Model Building

# Set seed for reproducibility
set.seed(20)

# Split the data into training and testing sets
train_index <- sample(1:nrow(data), 0.81 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Convert "Attrited Customer" to 1 and "Existing Customer" to 0
train_data$Attrition_Flag <- ifelse(train_data$Attrition_Flag == "Attrited Customer", 1, 0)
test_data$Attrition_Flag <- ifelse(test_data$Attrition_Flag == "Attrited Customer", 1, 0)

# Fit logistic regression model
log_model <- glm(Attrition_Flag ~ ., data = train_data, family = binomial)

# Summary of the logistic regression model
summary(log_model)

# Make predictions on the test data
predicted_values <- predict(log_model, newdata = test_data, type = "response")
predicted_classes <- ifelse(predicted_values > 0.5, 1, 0)

# Calculate accuracy
accuracy <- sum(predicted_classes == test_data$Attrition_Flag) / length(predicted_classes)
accuracy

# Decision Tree

# Fit decision tree model
library(rpart)
dt_model <- rpart(Attrition_Flag ~ ., data = train_data, method = "class")

# Make predictions using decision tree model
predicted_classes_dt <- predict(dt_model, newdata = test_data, type = "class")

# Calculate accuracy of decision tree model
accuracy_dt <- sum(predicted_classes_dt == test_data$Attrition_Flag) / length(predicted_classes_dt)
accuracy_dt
