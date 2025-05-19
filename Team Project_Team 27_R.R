###Import data###
data <- read.csv("C:/Users/chenx/OneDrive - Duke University/duke/Fall1/520/Team Project/movies.csv")

###Clean data###
data<- data[complete.cases(data),] #drop the missing data#
columns_with_na <- names(data)[colSums(is.na(data)) > 0]
columns_with_na #double check if there is any missing data#

data <- data[, !(names(data) %in% c("name", "released"))] #Remove features that are not useful for prediction(name/released)#

data <- subset(data, !(rating %in% c("Approved", "Not Rated", "Unrated"))) # Filter the dataframe to exclude the records with "Approved", "Not Rated", or "Unrated" in the "rating" column

top_directors <- names(sort(table(data$director), decreasing = TRUE))[1:10]
data$director <- ifelse(data$director %in% top_directors, data$director, "Other")
top_writer <- names(sort(table(data$writer), decreasing = TRUE))[1:10]
data$writer <- ifelse(data$writer %in% top_writer, data$writer, "Other")
top_star <- names(sort(table(data$star), decreasing = TRUE))[1:10]
data$star <- ifelse(data$star %in% top_star, data$star, "Other")
top_company <- names(sort(table(data$company), decreasing = TRUE))[1:10]
data$company <- ifelse(data$company %in% top_company, data$company, "Other") #keep the top 10 most frequent and encode all others into a category "Other."#

data$profit <- data$gross - data$budget ## Create a new column 'profit' by subtracting 'budget' from 'gross'#

#split data into training and test sets #
set.seed(123) # Set seed for reproducibility
sample_size <- floor(0.8 * nrow(data)) # Calculate the sample size (80% of the data)
train_indices <- sample(seq_len(nrow(data)), size = sample_size) # Randomly select rows for the training set
data_train <- data[train_indices, ] # Create the training set (80% of the data)
data_test <- data[-train_indices, ] # Create the test set (20% of the data)

###K- fold and models###
# Set up K-fold cross-validation (K=5)
library(ggplot2)
library(lattice)
library(caret)
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)

###Random forest Model###
library(randomForest)

# Training model for predicting "profit"
rf_model_profit <- train(profit ~ ., data = data_train, 
                         method = "rf", 
                         trControl = train_control, 
                         tuneLength = 3,  
                         ntree = 100)     
print(rf_model_profit)

# Training model for predicting "score"
rf_model_score <- train(score ~ ., data = data_train, 
                        method = "rf", 
                        trControl = train_control, 
                        tuneLength = 3,  
                        ntree = 100) 
print(rf_model_score) # Check model summary

# Predict "profit" on the test set
data_test <- data_test[!(data_test$country %in% c("Finland", "India", "Malta")), ] # Remove rows in the test set with unseen levels in the "country" column
pred_profit_rf <- predict(rf_model_profit, newdata = data_test)# Predict again on the cleaned test set

# Predict "score" on the test set
pred_score_rf <- predict(rf_model_score, newdata = data_test)

#Evaluating the Random forest Model#
mse_profit_rf <- mean((pred_profit_rf - data_test$profit)^2) # Mean Squared Error (MSE)
rmse_profit_rf <- sqrt(mse_profit_rf) # Root Mean Squared Error (RMSE)
mae_profit_rf <- mean(abs(pred_profit_rf - data_test$profit)) # Mean Absolute Error (MAE)

mse_score_rf <- mean((pred_score_rf - data_test$score)^2) # Mean Squared Error (MSE)
rmse_score_rf <- sqrt(mse_score_rf) # Root Mean Squared Error (RMSE)
mae_score_rf <- mean(abs(pred_score_rf - data_test$score)) # Mean Absolute Error (MAE)

# Out-of-Sample R-squared
rss_profit_rf <- sum((pred_profit_rf - data_test$profit)^2)   # Residual Sum of Squares
tss_profit_rf <- sum((data_test$profit - mean(data_test$profit))^2)  # Total Sum of Squares
r_squared_profit_rf <- 1 - (rss_profit_rf / tss_profit_rf)

rss_score_rf <- sum((pred_score_rf - data_test$score)^2)   # Residual Sum of Squares
tss_score_rf <- sum((data_test$score - mean(data_test$score))^2)  # Total Sum of Squares
r_squared_score_rf <- 1 - (rss_score_rf / tss_score_rf)

cat("Mean Squared Error (profit_rf):", mse_profit_rf, "\n")
cat("Root Mean Squared Error (profit_rf):", rmse_profit_rf, "\n")
cat("Mean Absolute Error (profit_rf):", mae_profit_rf, "\n")
cat("Out-of-Sample R-Squared for profit with rf:", r_squared_profit_rf, "\n")
cat("Mean Squared Error (score_rf):", mse_score_rf, "\n")
cat("Root Mean Squared Error (score_rf):", rmse_score_rf, "\n")
cat("Mean Absolute Error (score_rf):", mae_score_rf, "\n")
cat("Out-of-Sample R-Squared for score with rf:", r_squared_score_rf, "\n")

# Calculate residuals (difference between actual and predicted values)
residuals_profit_rf <- data_test$profit - pred_profit_rf
residuals_score_rf <- data_test$score - pred_score_rf
# Residual plot
ggplot(data.frame(pred_profit_rf, residuals_profit_rf), aes(x = pred_profit_rf, y = residuals_profit_rf)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residual Plot", x = "Predicted Profit", y = "Residuals")
ggplot(data.frame(pred_score_rf, residuals_score_rf), aes(x = pred_score_rf, y = residuals_score_rf)) +
  geom_point() +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Residual Plot", x = "Predicted Score", y = "Residuals")


###K-Nearest Neighbors (KNN)###
install.packages(glmnet)
library(glmnet)
library(dplyr)
library(class)

# Step 1: Create model matrix for the training set (excluding 'score', 'profit', and 'gross')
x_train <- model.matrix(~ . - score - profit - gross, data = data_train)
y_train_profit <- data_train$profit  # Target variable (profit)
y_train_score <- data_train$score  # Target variable (profit)

# Step 2: Create model matrix for the test set
x_test <- model.matrix(~ . - score - profit - gross, data = data_test)

# Step 3: Align columns of x_test with x_train
# Find columns that are in x_train but not in x_test
missing_cols <- setdiff(colnames(x_train), colnames(x_test))

# Add missing columns to x_test and fill them with 0 (because those categories are not present in test set)
for (col in missing_cols) {
  x_test <- cbind(x_test, setNames(data.frame(rep(0, nrow(x_test))), col))
}

# Step 4: Ensure both datasets have the same column order
x_test <- x_test[, colnames(x_train)]

# Step 5: Run KNN model for Profit Prediction
set.seed(123)
knn_profit <- train(x = x_train, y = y_train_profit, 
                    method = "knn", 
                    trControl = train_control, 
                    tuneLength = 5)
print(knn_profit)
# Predict profit on the test set
predicted_profit_knn <- predict(knn_profit, newdata = as.data.frame(x_test))


# For Profit
mse_profit_knn <- mean((predicted_profit_knn - data_test$profit)^2)
rmse_profit_knn <- sqrt(mse_profit_knn)
mae_profit_knn <- mean(abs(predicted_profit_knn - data_test$profit))

rss_profit_knn <- sum((predicted_profit_knn - data_test$profit)^2)   # Residual Sum of Squares
tss_profit_knn <- sum((data_test$profit - mean(data_test$profit))^2)  # Total Sum of Squares
r_squared_profit_knn <- 1 - (rss_profit_knn / tss_profit_knn)

cat("Mean Squared Error (profit_knn):", mse_profit_knn, "\n")
cat("Root Mean Squared Error (profit_knn):", rmse_profit_knn, "\n")
cat("Mean Absolute Error (profit_knn):", mae_profit_knn, "\n")
cat("Out-of-Sample R-Squared for profit with knn:", r_squared_profit_knn, "\n")

# Step 6: Run KNN model for Score Prediction
set.seed(123)
knn_score <- train(x = x_train, y = y_train_score, 
                    method = "knn", 
                    trControl = train_control, 
                    tuneLength = 5)
print(knn_score)
# Predict score on the test set
predicted_score_knn <- predict(knn_score, newdata = as.data.frame(x_test))

# For Score
mse_score_knn <- mean((predicted_score_knn - data_test$score)^2)
rmse_score_knn <- sqrt(mse_score_knn)
mae_score_knn <- mean(abs(predicted_score_knn - data_test$score))

rss_score_knn <- sum((predicted_score_knn - data_test$score)^2)   # Residual Sum of Squares
tss_score_knn <- sum((data_test$score - mean(data_test$score))^2)  # Total Sum of Squares
r_squared_score_knn <- 1 - (rss_score_knn / tss_score_knn)

cat("Mean Squared Error (score_knn):", mse_score_knn, "\n")
cat("Root Mean Squared Error (score_knn):", rmse_score_knn, "\n")
cat("Mean Absolute Error (score_knn):", mae_score_knn, "\n")
cat("Out-of-Sample R-Squared for score with knn:", r_squared_score_knn, "\n")


###post lasso###
# Load necessary libraries
library(glmnet)
library(caret)

# Step 1: Prepare the data (convert categorical variables if necessary)
# Convert character columns to factors
data_train_lm <- data_train %>%
  mutate(across(where(is.character), as.factor))

data_test_lm <- data_test %>%
  mutate(across(where(is.character), as.factor))

# Step 2: Apply one-hot encoding (convert factors into dummy variables)
# Create model matrix for training data (exclude 'gross' column)
x_train <- model.matrix(~ . - score - profit - gross, data = data_train_lm)
y_train_profit <- data_train$profit  # Target variable for profit
y_train_score <- data_train$score    # Target variable for score

# Create model matrix for test data (exclude 'gross' column)
x_test <- model.matrix(~ . - score - profit - gross, data = data_test_lm)

# Step 3: Identify features in training data that are not present in test data
# Find missing columns in x_test compared to x_train
missing_cols_in_test <- setdiff(colnames(x_train), colnames(x_test))

# Remove these columns from x_train to align it with x_test
x_train_aligned <- x_train[, !colnames(x_train) %in% missing_cols_in_test]

# Step 4: Apply Lasso with cross-validation to select features
set.seed(123)
train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

# Lasso for Profit Prediction (with cross-validation)
lasso_cv_profit <- cv.glmnet(x_train_aligned, y_train_profit, alpha = 1, nfolds = 5)

# Lasso for Score Prediction (with cross-validation)
lasso_cv_score <- cv.glmnet(x_train_aligned, y_train_score, alpha = 1, nfolds = 5)

# Step 5: Post-Lasso - Extract important features and fit linear regression models

# For Profit: Select important features from Lasso
lasso_coef_profit <- coef(lasso_cv_profit, s = "lambda.min")
important_features_profit <- rownames(lasso_coef_profit)[lasso_coef_profit[, 1] != 0]
important_features_profit <- important_features_profit[important_features_profit != "(Intercept)"]

cat("Important features selected by Lasso for profit:", important_features_profit, "\n")

# Fit a Linear Regression model using the selected features for profit
linear_model_profit <- lm(y_train_profit ~ ., data = as.data.frame(x_train_aligned[, important_features_profit]))

# For Score: Select important features from Lasso
lasso_coef_score <- coef(lasso_cv_score, s = "lambda.min")
important_features_score <- rownames(lasso_coef_score)[lasso_coef_score[, 1] != 0]
important_features_score <- important_features_score[important_features_score != "(Intercept)"]

cat("Important features selected by Lasso for score:", important_features_score, "\n")

# Fit a Linear Regression model using the selected features for score
linear_model_score <- lm(y_train_score ~ ., data = as.data.frame(x_train_aligned[, important_features_score]))

# Step 6: Predict on the test set (make sure columns match)
x_test_aligned <- x_test[, colnames(x_train_aligned)]

# Predict Profit using Post-Lasso Linear Regression
predicted_profit_lm <- predict(linear_model_profit, newdata = as.data.frame(x_test_aligned[, important_features_profit]))

# Predict Score using Post-Lasso Linear Regression
predicted_score_lm <- predict(linear_model_score, newdata = as.data.frame(x_test_aligned[, important_features_score]))

# Step 7: Evaluate model performance (e.g., RMSE and R-squared)

# For Profit
mse_profit_lm <- mean((predicted_profit_lm - data_test$profit)^2)
rmse_profit_lm <- sqrt(mse_profit_lm)
mae_profit_lm <- mean(abs(predicted_profit_lm - data_test$profit))

rss_profit_lm <- sum((predicted_profit_lm - data_test$profit)^2)   # Residual Sum of Squares
tss_profit_lm <- sum((data_test$profit - mean(data_test$profit))^2)  # Total Sum of Squares
r_squared_profit_lm <- 1 - (rss_profit_lm / tss_profit_lm)

cat("Mean Squared Error (profit_lm):", mse_profit_lm, "\n")
cat("Root Mean Squared Error (profit_lm):", rmse_profit_lm, "\n")
cat("Mean Absolute Error (profit_lm):", mae_profit_lm, "\n")
cat("Out-of-Sample R-Squared for profit with lm:", r_squared_profit_lm, "\n")


# For Score
mse_score_lm <- mean((predicted_score_lm - data_test$score)^2)
rmse_score_lm <- sqrt(mse_score_lm)
mae_score_lm <- mean(abs(predicted_score_lm - data_test$score))

rss_score_lm <- sum((predicted_score_lm - data_test$score)^2)   # Residual Sum of Squares
tss_score_lm <- sum((data_test$score - mean(data_test$score))^2)  # Total Sum of Squares
r_squared_score_lm <- 1 - (rss_score_lm / tss_score_lm)

cat("Mean Squared Error (score_lm):", mse_score_lm, "\n")
cat("Root Mean Squared Error (score_lm):", rmse_score_lm, "\n")
cat("Mean Absolute Error (score_lm):", mae_score_lm, "\n")
cat("Out-of-Sample R-Squared for score with lm:", r_squared_score_lm, "\n")     

###Evaluation###
library(ggplot2)
library(reshape2)

# Step 1: Create a data frame with performance metrics for each model
model_comparison_score <- data.frame(
  Model = c("Linear Regression", "Random Forest", "KNN"),
  MSE = c(mse_score_lm, mse_score_rf, mse_score_knn),
  RMSE = c(rmse_score_lm, rmse_score_rf, rmse_score_knn),
  MAE = c(mae_score_lm, mae_score_rf, mae_score_knn),
  R_squared = c(r_squared_score_lm, r_squared_score_rf, r_squared_score_knn)
)
model_comparison_profit <- data.frame(
  Model = c("Linear Regression", "Random Forest", "KNN"),
  MSE = c(mse_profit_lm, mse_profit_rf, mse_profit_knn),
  RMSE = c(rmse_profit_lm, rmse_profit_rf, rmse_profit_knn),
  MAE = c(mae_profit_lm, mae_profit_rf, mae_profit_knn),
  R_squared = c(r_squared_profit_lm, r_squared_profit_rf, r_squared_profit_knn)
)
# Step 2: Melt the data frame for easy plotting with ggplot
model_comparison_melted_score <- melt(model_comparison_score, id.vars = "Model")
model_comparison_melted_profit <- melt(model_comparison_profit, id.vars = "Model")

# Step 3: Plot the performance metrics using ggplot
ggplot(model_comparison_melted_score, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  theme_minimal() +
  labs(title = "Comparison of Model Performance Metrics",
       x = "Model", y = "Metric Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(model_comparison_melted_profit, aes(x = Model, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ variable, scales = "free", ncol = 2) +
  theme_minimal() +
  labs(title = "Comparison of Model Performance Metrics",
       x = "Model", y = "Metric Value") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
