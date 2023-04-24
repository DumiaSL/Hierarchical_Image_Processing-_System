
# Load necessary libraries
library(readxl)

#part 1

# Load the UOW consumption dataset
uow_consumption_dataset <- read_xlsx("data sets/uow_consumption.xlsx")


# Extract the hourly electricity consumption data for 20:00 for 2018 and 2019
hourly_consumption_20 <- uow_consumption_dataset[c("date", "0.83333333333333337")]

print(hourly_consumption_20)

# Extract the first 380 samples as training data, and the remaining samples as testing data
train_data <- unlist(hourly_consumption_20[1:380, "0.83333333333333337"])
test_data <- unlist(hourly_consumption_20[381:nrow(hourly_consumption_20), "0.83333333333333337"])

# Define the number of time-delayed inputs
num_inputs <- 24

print(train_data)

print(length(train_data))

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


# Print the input/output matrix
print(input_output_matrix)