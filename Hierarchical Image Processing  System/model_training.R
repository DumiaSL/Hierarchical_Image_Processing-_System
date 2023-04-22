# Load necessary libraries
library(readxl)
library(datasets)
library(NbClust)

# Load the vehicles dataset
vehicles_data <- read_xlsx("data sets/vehicles.xlsx")

# Remove the output variable (class attribute)
vehicles_data <- vehicles_data[, 1:18]

# Detect and remove outliers using the IQR method
q1 <- apply(vehicles_data, 2, quantile, 0.25)
q3 <- apply(vehicles_data, 2, quantile, 0.75)
iqr <- q3 - q1
threshold <- 1.5 * iqr
outliers <- apply(vehicles_data, 2, function(x) x < (q1 - threshold) | x > (q3 + threshold))
vehicles_data <- vehicles_data[rowSums(outliers) == 0,]

# Scale the data using standardization
scaled_vehicles_data <- scale(vehicles_data)

# Use NbClust to determine the number of clusters
nb_clusters <- NbClust(scaled_vehicles_data, min.nc = 2, max.nc = 5, method = "kmeans", index = "silhouette")

# Print the best number of clusters
cat("Best number of clusters based on automated methods: ", nb_clusters$Best.nc, "\n")

# Perform k-means clustering using the most favoured number of clusters
num_clusters <- nb_clusters$Best.nc
set.seed(123)
kmeans_result <- kmeans(scaled_vehicles_data, centers = matrix(rnorm(num_clusters * ncol(scaled_vehicles_data)), ncol = ncol(scaled_vehicles_data)), nstart = 25)

# Print the kmeans output
cat("kmeans output:\n")
print(kmeans_result)

# Calculate between-cluster sum of squares (BSS), total sum of squares (TSS), and within-cluster sum of squares (WSS) indices
BSS <- sum((colMeans(scaled_vehicles_data) - colMeans(kmeans_result$centers))^2) * nrow(scaled_vehicles_data)
TSS <- sum(apply(scaled_vehicles_data, 2, function(x) sum((x - mean(x))^2)))
WSS <- sum(kmeans_result$withinss)

# Print BSS/TSS ratio
cat("BSS/TSS ratio:", BSS/TSS, "\n")

# Plot the clustering results
plot(scaled_vehicles_data, col = kmeans_result$cluster)
points(kmeans_result$centers, col = 1:num_clusters, pch = 8, cex = 2)
