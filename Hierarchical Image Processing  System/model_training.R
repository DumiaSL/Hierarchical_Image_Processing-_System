# Load necessary libraries
library(readxl)
library(ggplot2)
library(NbClust)
library(cluster)
library(FactoMineR)
library(factoextra)
library(fpc)

#part 1

# Load the vehicles dataset
vehicles_data <- read_xlsx("data sets/vehicles.xlsx")

# Remove the output variable (class attribute)
vehicles_data <- vehicles_data[, 1:18]

# Scale the data using standardization
scaled_vehicles_data <- scale(vehicles_data)

# Detect and remove outliers using the IQR method
q1 <- apply(scaled_vehicles_data, 2, quantile, 0.25)
q3 <- apply(scaled_vehicles_data, 2, quantile, 0.75)
iqr <- q3 - q1
threshold <- 1.5 * iqr
outliers <- apply(scaled_vehicles_data, 2, function(x) x < (q1 - threshold) | x > (q3 + threshold))
scaled_vehicles_data <- scaled_vehicles_data[rowSums(outliers) == 0,]


# Perform PCA on the scaled data
pca <- prcomp(scaled_vehicles_data, scale = TRUE)

# Extract the first two principal components
pc1 <- pca$x[, 1]
pc2 <- pca$x[, 2]

# Create a scatter plot of the first two principal components
ggplot(data.frame(pc1, pc2), aes(x = pc1, y = pc2)) +
  geom_point() +
  labs(x = "Principal Component 1", y = "Principal Component 2",
       title = "Scatter plot of first two principal components")


#--------------------------------------------------------------------------------------------------------------
# Use NbClust to determine the number of clusters in scaled_vehicles_data
nb_clusters <- NbClust(scaled_vehicles_data, min.nc = 2, max.nc = 5, method = "kmeans", index = "silhouette")

# Plot the bar plot of the clustering indices
par(mar=c(1,1,1,1))
plot(nb_clusters$All.index, type = "b", xlab = "Number of clusters", ylab = "Clustering index", main = "NbClust plot")
abline(v = nb_clusters$Best.nc, col = "blue")

# Print the best number of clusters in scaled_vehicles_data
cat("Best number of clusters based on NbClust automated methods: ", nb_clusters$Best.nc, "\n")


#--------------------------------------------------------------------------------------------------------------
# Create an elbow curve for KMeans clustering
# Calculate the within-cluster sum of squares for different values of k
wcss <- vector("numeric", length = 5)
for (i in 2:5) {
  kmeans_model <- kmeans(scaled_vehicles_data, centers = i, nstart = 25)
  wcss[i] <- kmeans_model$tot.withinss
}

# Plot the elbow curve
plot(1:5, wcss, type = "b", xlab = "Number of clusters", ylab = "WCSS")
title(main = "Elbow curve for k-means clustering")
abline(v = 3, col = "red", lty = 2)

# Find the "elbow" in the plot
diffs <- diff(wcss)
elbow <- which(diffs == min(diffs)) + 1

# Print the best number of clusters based on the elbow method
cat("Best number of clusters based on the elbow method: ", elbow, "\n")


#--------------------------------------------------------------------------------------------------------------
# Calculate the gap statistic for different values of k
set.seed(123)
gap_stat <- clusGap(scaled_vehicles_data, kmeans, nstart = 25, K.max = 10, B = 50)


# Plot the gap statistic
plot(gap_stat, main = "Gap statistic for k-means clustering")

# Identify the optimal number of clusters
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")
cat("Optimal number of clusters based on the gap statistic: ", optimal_k, "\n")


#--------------------------------------------------------------------------------------------------------------
# Calculate the average silhouette width for different values of k
# Set the range of K values to test
k.min <- 2
k.max <- 10

# Create a list to store the silhouette values for each value of K
silhouette_vals <- vector("list", k.max - k.min + 1)

# Loop through each value of K and perform clustering using K-means algorithm
for (k in k.min:k.max) {
  km <- kmeans(scaled_vehicles_data, centers = k, nstart = 10)
  
  # Calculate the silhouette width for each data point
  silhouette_vals[[k - k.min + 1]] <- silhouette(km$cluster, dist(scaled_vehicles_data))
}

# Calculate the average silhouette width for each value of K
silhouette_avg <- sapply(silhouette_vals, function(x) mean(x[, 3]))

# Plot the silhouette widths for each value of K
plot(k.min:k.max, silhouette_avg, type = "b", xlab = "Number of clusters", ylab="Silhouette")

# Find the index of the maximum silhouette width
best_k <- which.max(silhouette_avg) + k.min - 1

# Print the best number of clusters based on the silhouette method
cat("Best number of clusters based on the silhouette method: ", best_k, "\n")


#--------------------------------------------------------------------------------------------------------------
# Check if multiple optimal numbers of clusters were found in scaled_vehicles_data
if (length(nb_clusters$Best.nc) > 1) {
  num_clusters <- nb_clusters$Best.nc[1]
  cat("Multiple optimal numbers of clusters found, using the first one:", num_clusters, "\n")
} else {
  num_clusters <- nb_clusters$Best.nc
}

# Perform k-means clustering using the most favoured number of clusters in scaled_vehicles_data
if(num_clusters > 1) {
  set.seed(123)
  kmeans_result <- kmeans(scaled_vehicles_data, centers = matrix(rnorm(num_clusters * ncol(scaled_vehicles_data)), ncol = ncol(scaled_vehicles_data)), nstart = 25)
  
  # Print the kmeans output in scaled_vehicles_data
  cat("kmeans output:\n")
  print(kmeans_result)
  
  # Calculate between-cluster sum of squares (BSS), total sum of squares (TSS), and within-cluster sum of squares (WSS) indices
  BSS <- sum((colMeans(scaled_vehicles_data) - colMeans(kmeans_result$centers))^2) * nrow(scaled_vehicles_data)
  TSS <- sum(apply(scaled_vehicles_data, 2, function(x) sum((x - mean(x))^2)))
  WSS <- sum(kmeans_result$withinss)
  
  # Print BSS/TSS ratio
  cat("BSS/TSS ratio:", BSS/TSS, "\n")
  cat("BSS_indices : ",BSS, "\n")
  cat("WSS_indices : ",WSS, "\n")
  
  # Plot the clustering results
  plot(scaled_vehicles_data, col = kmeans_result$cluster)
  points(kmeans_result$centers, col = 1:num_clusters, pch = 8, cex = 2)
  
  # Calculate silhouette coefficients and plot the silhouette plot
  silhouette_obj <- silhouette(kmeans_result$cluster, dist(scaled_vehicles_data))
  plot(silhouette_obj)
} else {
  cat("Cannot perform k-means clustering with only one cluster.\n")
}

#------------------------------------------------------------------------------------------------------------------------------
#part 2

# Apply PCA analysis
pca_result <- prcomp(scaled_vehicles_data, center = TRUE, scale. = TRUE)

# Print eigenvalues and eigenvectors
cat("Eigenvalues:\n")
print(pca_result$eig)
cat("Eigenvectors:\n")
print(pca_result$var)

# Calculate cumulative score per principal components
cumulative_score <- cumsum(pca_result$eig/sum(pca_result$eig) * 100)

# Print the cumulative scores
cat("Cumulative scores:\n")
print(cumulative_score)

# Choose the number of principal components that provide at least cumulative score > 92%
num_pcs <- length(cumulative_score[cumulative_score <= 92]) + 1

# Create a new transformed dataset with principal components as attributes
transformed_vehicles_data <- predict(pca_result, newdata = scaled_vehicles_data)[, 1:num_pcs]

# Print the transformed dataset
cat("Transformed dataset:\n")
print(transformed_vehicles_data)


#--------------------------------------------------------------------------------------------------------------
# Use NbClust to determine the number of clusters in transformed_vehicles_data
nb_clusters_transf <- NbClust(transformed_vehicles_data, min.nc = 2, max.nc = 5, method = "kmeans", index = "silhouette")

# Print the best number of clusters in transformed_vehicles_data
cat("Best number of clusters in transformed_vehicles_data based on automated methods: ", nb_clusters_transf$Best.nc, "\n")


#--------------------------------------------------------------------------------------------------------------
# Create an elbow curve for KMeans clustering
# Calculate the within-cluster sum of squares for different values of k
wcss <- vector("numeric", length = 5)
for (i in 2:5) {
  kmeans_model <- kmeans(transformed_vehicles_data, centers = i, nstart = 25)
  wcss[i] <- kmeans_model$tot.withinss
}

# Plot the elbow curve
plot(1:5, wcss, type = "b", xlab = "Number of clusters ", ylab = "WCSS")
title(main = "Elbow curve for k-means clustering")
abline(v = 3, col = "red", lty = 2)

# Find the "elbow" in the plot
diffs <- diff(wcss)
elbow <- which(diffs == min(diffs)) + 1

# Print the best number of clusters based on the elbow method
cat("Best number of clusters in transformed_vehicles_data based on the elbow method: ", elbow, "\n")

# Reshape the vector "transformed_vehicles_data" into a matrix with a single column
transformed_vehicles_data <- matrix(transformed_vehicles_data, ncol = 1)


#--------------------------------------------------------------------------------------------------------------
# Calculate the gap statistic for different values of k
set.seed(123)
gap_stat <- clusGap(transformed_vehicles_data, kmeans, nstart = 25, K.max = 10, B = 50)

# Plot the gap statistic
plot(gap_stat, main = "Gap statistic for k-means clustering")

# Identify the optimal number of clusters
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")
cat("Optimal number of clusters based on the gap statistic: ", optimal_k, "\n")


#--------------------------------------------------------------------------------------------------------------
# Calculate the between_cluster_sums_of_squares (BSS) and the total_sum_of_squares (TSS)

# Perform k-means clustering on the transformed data with the best k value
set.seed(123)
kmeans_transf <- kmeans(transformed_vehicles_data, nb_clusters_transf$Best.nc)

# Plot the clustering results
plot(transformed_vehicles_data, col = kmeans_transf$cluster)
points(kmeans_transf$centers, col = 1:nb_clusters_transf$Best.nc, pch = 8, cex = 2)

# Print the k-means output
print(kmeans_transf)

# Calculate BSS and WSS for transformed_vehicles_data
BSS_transf <- sum(kmeans_transf$size * apply((kmeans_transf$centers - apply(transformed_vehicles_data, 2, mean))^2, 1, sum))
TSS_transf <- sum(apply(transformed_vehicles_data^2, 1, sum)) - (sum(transformed_vehicles_data)^2)/length(transformed_vehicles_data)
WSS_transf <- kmeans_transf$tot.withinss

# Print BSS/TSS ratio, BSS, and WSS for transformed_vehicles_data
cat("BSS/TSS ratio for transformed_vehicles_data: ", BSS_transf/TSS_transf, "\n")
cat("BSS for transformed_vehicles_data: ", BSS_transf, "\n")
cat("WSS for transformed_vehicles_data: ", WSS_transf, "\n")

# Print k-means output for transformed_vehicles_data
print(kmeans_transf)


# Plot the clustering results
plot(transformed_vehicles_data, col = kmeans_transf$cluster)
points(kmeans_transf$centers, col = 1:num_clusters, pch = 8, cex = 2)

# Calculate silhouette coefficients and plot the silhouette plot
silhouette_obj <- silhouette(kmeans_transf$cluster, dist(transformed_vehicles_data))
plot(silhouette_obj)

# Calculating Average silhouette width score
silhouette_avg <- mean(silhouette_obj[, 3])
cat("Average silhouette width score: ", silhouette_avg, "\n")

# Compute Calinski-Harabasz index
ch_index <- cluster.stats(transformed_vehicles_data, kmeans_transf$cluster)[[1]]

# Print the CH index
cat("Calinski-Harabasz Index: ", ch_index, "\n")