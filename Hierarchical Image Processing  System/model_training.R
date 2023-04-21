#Load the data set - loading the vehicles exel file 
library(readxl)
data <- read_xlsx("data sets/vehicles.xlsx")

#Remove the Class attribute (output variable) - remove the vehical lables
data <- data[, 1:18]


#Detect and remove outliers using the IQR method
q1 <- apply(data, 2, quantile, 0.25)
q3 <- apply(data, 2, quantile, 0.75)
iqr <- q3 - q1
threshold <- 1.5 * iqr
outliers <- apply(data, 2, function(x) x < (q1 - threshold) | x > (q3 + threshold))
data <- data[rowSums(outliers) == 0,]

#Scale the data using standardization
scaled_data <- scale(data)
