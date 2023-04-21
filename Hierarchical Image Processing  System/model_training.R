#Load the data set - loading the vehicles exel file 
library(readxl)
data <- read_xlsx("data sets/vehicles.xlsx")

#Remove the Class attribute (output variable) - remove the vehical lables
data <- data[, 1:18]

