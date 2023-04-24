# Load necessary libraries
library(readxl)
library(neuralnet)
library(MLmetrics)

# Define the root-mean-square error (RMSE) function
rmse <- function(error) {
  return(sqrt(mean(error^2)))
}

# Define the mean absolute error (MAE) function
mae <- function(error) {
  return(mean(abs(error)))
}


# Load the UOW consumption dataset
uow_consumption_dataset <- read_xlsx("data sets/uow_consumption.xlsx")

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


# Train the MLP using the normalized input/output matrix
mlp <- neuralnet(V2 ~ ., data=input_output_matrix, hidden=c(5,2), linear.output=TRUE)

#Plot the neural network
plot(mlp)

# Extract the inputs for the test data
test_inputs <- matrix(test_data[1:(length(test_data)-num_inputs)], ncol=num_inputs, byrow=TRUE)

# Predict the output values for the test data
mlp_output <- predict(mlp, test_inputs)

# Denormalize the predicted output values
mlp_output <- (mlp_output * sd(train_data)) + mean(train_data)

# Calculate the MAE for the predicted output values and the actual output values
mae_result <- mae(mlp_output - test_data[(num_inputs+1):length(test_data)])

# Print the MAE result
cat("The MAE for the test data is:", round(mae_result, 2))

# Calculate the RMSE for the predicted output values and the actual output values
rmse_result <- rmse(mlp_output - test_data[(num_inputs+1):length(test_data)])

# Print the RMSE result
cat("The RMSE for the test data is:", round(rmse_result, 2))

# Define the mean absolute percentage error (MAPE) function
mape <- function(actual, predicted) {
  return(mean(abs((actual - predicted)/actual)) * 100)
}

# Calculate the MAPE for the predicted output values and the actual output values
mape_result <- mape(test_data[(num_inputs+1):length(test_data)], mlp_output)

# Print the MAPE result
cat("The MAPE for the test data is:", round(mape_result, 2))

# Define the symmetric mean absolute percentage error (sMAPE) function
smape <- function(actual, predicted) {
  return(2 * mean(abs(actual - predicted) / (abs(actual) + abs(predicted))) * 100)
}

# Calculate the sMAPE for the predicted output values and the actual output values
smape_result <- smape(test_data[(num_inputs+1):length(test_data)], mlp_output)

# Print the sMAPE result
cat("The sMAPE for the test data is:", round(smape_result, 2))
