# Load necessary libraries
library(readxl)
library(neuralnet)
library(MLmetrics)
library(keras)

#part 1

# Define the root-mean-square error (RMSE) function
rmse <- function(error) {
  return(sqrt(mean(error^2)))
}

# Define the mean absolute error (MAE) function
mae <- function(error) {
  return(mean(abs(error)))
}

# Define the mean absolute percentage error (MAPE) function
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted)/actual)) * 100)
}

# Define the symmetric mean absolute percentage error (sMAPE) function
smape <- function(actual, predicted) {
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
}

# Load the UOW consumption dataset
uow_consumption_dataset <- read_xlsx("data sets/uow_consumption.xlsx")

# summary(uow_consumption_dataset) # Get summary statistics for the column


# Extract the hourly electricity consumption data for 20:00 for 2018 and 2019
#   "0.83333333333333337"= 20.00  (20/24 = 0.83333333333333337)
hourly_consumption_20 <- uow_consumption_dataset[c("date", "0.83333333333333337")]

# Extract the first 380 samples as training data, and the remaining samples as testing data
train_data <- unlist(hourly_consumption_20[1:380, "0.83333333333333337"])
test_data <- unlist(hourly_consumption_20[381:nrow(hourly_consumption_20), "0.83333333333333337"])

# Define the number of time-delayed inputs
num_inputs <- 60

# Check that num_inputs is not larger than the length of train_data
if (num_inputs >= length(train_data)) {
  stop("The number of time-delayed inputs is too large for the length of the training data.")
}

# Construct the input/output matrix for MLP training/testing
input_output_matrix <- matrix(0, nrow=length(train_data)-num_inputs, ncol=num_inputs+1)

for (i in 1:(length(train_data)-num_inputs)) {
  input_output_matrix[i, 1:num_inputs] <- train_data[i:(i+num_inputs-1)]
  input_output_matrix[i, num_inputs+1] <- train_data[i+num_inputs]
}

# Normalize the input/output matrix
input_output_matrix <- apply(input_output_matrix, 2, function(x) (x - mean(x)) / sd(x))

# Define the neural network structures to be evaluated
structures <- list(
  c(5),
  c(10),
  c(5, 3),
  c(10, 5),
  c(10, 5, 3),
  c(20),
  c(20, 10),
  c(20, 10, 5),
  c(30),
  c(30, 20),
  c(30, 20, 10),
  c(50),
  c(50, 30),
  c(50, 30, 20)
)

results <- list()

for (i in 1:length(structures)) {

  # Train the MLP using the normalized input/output matrix
  mlp <- neuralnet(V2 ~ ., data=input_output_matrix, hidden=structures[[i]], linear.output=TRUE)

  # #Plot the neural network
  # plot(mlp)

  # Extract the inputs for the test data
  test_inputs <- matrix(test_data[1:(length(test_data)-num_inputs)], ncol=num_inputs, byrow=TRUE)

  # Predict the output values for the test data
  mlp_output <- predict(mlp, test_inputs)

  # Denormalize the predicted output values
  mlp_output <- (mlp_output * sd(train_data)) + mean(train_data)

  # Calculate the MAE for the predicted output values and the actual output values
  mae_result <- mae(mlp_output - test_data[(num_inputs+1):length(test_data)])

  cat("the test performances for c(",structures[[i]],")\n")

  # Print the MAE result
  cat("The MAE for the test data is:", round(mae_result, 2),"\n")

  # Calculate the RMSE for the predicted output values and the actual output values
  rmse_result <- rmse(mlp_output - test_data[(num_inputs+1):length(test_data)])

  # Print the RMSE result
  cat("The RMSE for the test data is:", round(rmse_result, 2),"\n")

  # Define the mean absolute percentage error (MAPE) function
  mape <- function(actual, predicted) {
    return(mean(abs((actual - predicted)/actual)) * 100)
  }

  # Calculate the MAPE for the predicted output values and the actual output values
  mape_result <- mape(test_data[(num_inputs+1):length(test_data)], mlp_output)

  # Print the MAPE result
  cat("The MAPE for the test data is:", round(mape_result, 2),"\n")

  # Define the symmetric mean absolute percentage error (sMAPE) function
  smape <- function(actual, predicted) {
    return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
  }

  # Calculate the sMAPE for the predicted output values and the actual output values
  smape_result <- smape(test_data[(num_inputs+1):length(test_data)], mlp_output)

  # Print the sMAPE result
  cat("The sMAPE for the test data is:", round(smape_result, 2),"\n\n")

  # Store the results for the current neural network structure
  results[[i]] <- c(structures[[i]], mae_result, rmse_result, mape_result, smape_result)

}

# Create a data frame of the results
results_df <- data.frame(matrix(unlist(results), ncol=5, byrow=TRUE))
colnames(results_df) <- c("Structure", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)")

# Print the comparison table of testing performances
print(results_df)

# Find the best one-hidden and two-hidden layer structures based on MAE and total number of weights
best_one_hidden <- results_df[which.min(results_df$MAE & results_df$Structure),]
best_two_hidden <- results_df[which.min(results_df$MAE & results_df$Structure),]

# Print the results
cat("Based on the comparison table, the best one-hidden layer neural network structure is",
    paste0("c(", best_one_hidden$Structure, ")"),
    "with a MAE of", best_one_hidden$MAE,
    "and a total number of", best_one_hidden$Structure + 1, "*1+1*1=", best_one_hidden$Structure + 2, "weight parameters.\n")
cat("The best two-hidden layer neural network structure is",
    paste0("c(", best_two_hidden$Structure, ")"),
    "with a MAE of", best_two_hidden$MAE,
    "and a total number of", sum(best_two_hidden$Structure) + length(best_two_hidden$Structure) + 1, "*1+1*1=", sum(best_two_hidden$Structure) + length(best_two_hidden$Structure) + 2, "weight parameters.\n")

#part 2

# Extract the hourly electricity consumption data for 18:00 , 19:00 for 2018 and 2019
#   "0.75"= 18.00   0.7916666666666667=19.00
hourly_consumption_18_19 <- uow_consumption_dataset[c("date", "0.75", "0.79166666666666663")]

# Extract the first 380 samples as training data, and the remaining samples as testing data
train_data_18_19 <- unlist(hourly_consumption_18_19[1:380, c("0.75","0.79166666666666663")])
test_data_18_19 <- unlist(hourly_consumption_18_19[381:nrow(hourly_consumption_18_19), c("0.75", "0.79166666666666663")])

# Define the number of time-delayed inputs
num_inputs_18_19 <- 60


# Check that num_inputs is not larger than the length of train_data
if (num_inputs_18_19 >= length(train_data_18_19)) {
  stop("The number of time-delayed inputs is too large for the length of the training data.")
}

# Construct the input/output matrix for MLP training/testing
input_output_matrix_18_19 <- matrix(0, nrow=length(train_data_18_19)-num_inputs_18_19, ncol=num_inputs_18_19+1)

for (i in 1:(length(train_data_18_19)-num_inputs_18_19)) {
  input_output_matrix_18_19[i, 1:num_inputs_18_19] <- train_data_18_19[i:(i+num_inputs_18_19-1)]
  input_output_matrix_18_19[i, num_inputs_18_19+1] <- train_data_18_19[i+num_inputs_18_19]
}

# Normalize the input/output matrix
input_output_matrix_18_19 <- apply(input_output_matrix_18_19, 2, function(x) (x - mean(x)) / sd(x))

# Define the NARX model architecture
narx_model <- neuralnet(V1 ~ ., data=input_output_matrix_18_19, hidden=c(5, 3), 
                        linear.output=TRUE, err.fct="sse", act.fct="logistic",
                        threshold=0.01, stepmax=1e+05, lifesign="full", 
                        algorithm="rprop+", learningrate=0.01)

# Plot the NARX model architecture
plot(narx_model, rep="best")


# Define the neural network structures to be evaluated
structures <- list(
  c(5),
  c(5, 3),
  c(10, 5, 3),
  c(20, 10),
  c(30),
  c(30, 20, 10)
)

results <- list()

for (i in 1:length(structures)) {
  
  # Prepare the input/output data for NARX approach
  narx_data <- buildTmatrix(train_data_18_19, num_inputs, num_outputs, horizon)
  
  # Split the input/output data into training and testing sets for NARX approach
  narx_splits <- splitTrainTest(narx_data, train_perc)
  
  # Train the NARX neural network using the training data
  narx_net <- narxnet(narx_splits$train$input, narx_splits$train$output, sizes=structures[[i]])
  
  # Make predictions for the testing data using the NARX neural network
  narx_predictions <- predict.narx(narx_net, narx_splits$test$input, n.ahead=horizon)
  
  # Denormalize the predicted output values
  narx_output <- (narx_predictions * sd(train_data_18_19[,2])) + mean(train_data[,2])
  
  # Calculate the MAE for the predicted output values and the actual output values
  mae_result <- mae(narx_output - test_data_18_19[(num_inputs+1):length(test_data_18_19)])
  
  cat("the test performances for c(",structures[[i]],")\n")
  
  # Print the MAE result
  cat("The MAE for the test data is:", round(mae_result, 2),"\n")
  
  # Calculate the RMSE for the predicted output values and the actual output values
  rmse_result <- rmse(narx_output - test_data_18_19[(num_inputs+1):length(test_data_18_19)])
  
  # Print the RMSE result
  cat("The RMSE for the test data is:", round(rmse_result, 2),"\n")
  
  # Define the mean absolute percentage error (MAPE) function
  mape <- function(actual, predicted) {
    return(mean(abs((actual - predicted)/actual)) * 100)
  }
  
  # Calculate the MAPE for the predicted output values and the actual output values
  mape_result <- mape(test_data_18_19[(num_inputs+1):length(test_data_18_19)], narx_output)
  
  # Print the MAPE result
  cat("The MAPE for the test data is:", round(mape_result, 2),"\n")
  
  # Define the symmetric mean absolute percentage error (sMAPE) function
  smape <- function(actual, predicted) {
    return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
  }
  
  # Calculate the sMAPE for the predicted output values and the actual output values
  smape_result <- smape(test_data_18_19[(num_inputs+1):length(test_data_18_19)], narx_output)
  
  # Print the sMAPE result
  cat("The sMAPE for the test data is:", round(smape_result, 2),"\n\n")
  
  # Store the results for the current NARX neural network structure
  results[[i]] <- c(structures[[i]], mae_result, rmse_result, mape_result, smape_result)
}

# Create a data frame of the results
results_df <- data.frame(matrix(unlist(results), ncol=5, byrow=TRUE))
colnames(results_df) <- c("Structure", "MAE", "RMSE", "MAPE (%)", "sMAPE (%)")

# Print the comparison table of testing performances
print(results_df)

# Find the best one-hidden and two-hidden layer structures based on MAE and total number of weights
best_one_hidden <- results_df[which.min(results_df$MAE & results_df$Structure),]
best_two_hidden <- results_df[which.min(results_df$MAE & results_df$Structure),]

# Print the results
cat("Based on the comparison table, the best one-hidden layer neural network structure is",
    paste0("c(", best_one_hidden$Structure, ")"),
    "with a MAE of", best_one_hidden$MAE,
    "and a total number of", best_one_hidden$Structure + 1, "*1+1*1=", best_one_hidden$Structure + 2, "weight parameters.\n")
cat("The best two-hidden layer neural network structure is",
    paste0("c(", best_two_hidden$Structure, ")"),
    "with a MAE of", best_two_hidden$MAE,
    "and a total number of", sum(best_two_hidden$Structure) + length(best_two_hidden$Structure) + 1, "*1+1*1=", sum(best_two_hidden$Structure) + length(best_two_hidden$Structure) + 2, "weight parameters.\n")
