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

# Calculate z-scores for each variable
z_scores <- apply(scaled_vehicles_data, 2, function(x) abs(scale(x, center = TRUE, scale = FALSE)))

# Identify rows with any z-score greater than 3 (a common threshold for outliers)
outliers <- apply(z_scores, 1, max) > 3

# Remove outlier rows from the dataset
scaled_vehicles_data <- scaled_vehicles_data[!outliers,]


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
set.seed(123)
par(mar=c(1,1,1,1))
nb_clusters <- NbClust(scaled_vehicles_data, min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

# Create a data frame with the clustering indices and the number of clusters
df <- data.frame(Clusters = 2:10, nb_clusters$All.index)

# Melt the data frame to long format
df_long <- reshape2::melt(df, id.vars = "Clusters", variable.name = "Index", value.name = "Value")

# Plot the bar plot using ggplot2
ggplot(df_long, aes(x = Clusters, y = Value, fill = Index)) +
  geom_bar(stat = "identity", position = "dodge") +
  xlab("Number of clusters") +
  ylab("Clustering index") +
  ggtitle("NbClust plot") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_vline(xintercept = nb_clusters$Best.nc[1], linetype = "dashed", color = "blue")

#--------------------------------------------------------------------------------------------------------------
# Create an elbow curve for KMeans clustering
# Calculate the within-cluster sum of squares for different values of k
wcss <- vector("numeric", length = 5)
for (i in 2:5) {
  kmeans_model <- kmeans(scaled_vehicles_data, centers = i, nstart = 25)
  wcss[i] <- kmeans_model$tot.withinss
}

# Plot the elbow curve
# Create a data frame with the WCSS values and the number of clusters
df <- data.frame(Clusters = 1:5, WCSS = wcss)

# Plot the elbow curve using ggplot2
ggplot(df, aes(x = Clusters, y = WCSS)) +
  geom_point() +
  geom_line() +
  xlab("Number of clusters") +
  ylab("WCSS") +
  ggtitle("Elbow curve for k-means clustering") +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red")

# Find the "elbow" in the plot
diffs <- diff(wcss)
elbow <- which(diffs == min(diffs)) + 1

# Print the best number of clusters based on the elbow method
cat("Best number of clusters based on the elbow method: ", elbow, "\n")


#--------------------------------------------------------------------------------------------------------------
# Calculate the gap statistic for different values of k
#Gap statistics
gap_stat <- clusGap(scaled_vehicles_data, FUN = kmeans, nstart = 25,
                    K.max = 3, B = 50)
# Convert gap_stat object to a data frame
gap_df <- data.frame(
  k = 1:3,
  gap = gap_stat$Tab[, "gap"],
  se = gap_stat$Tab[, "SE.sim"]
)

# Plot the Gap Statistic using ggplot2
ggplot(gap_df, aes(x = k, y = gap)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = gap - se, ymax = gap + se), width = 0.2) +
  labs(x = "Number of Clusters", y = "Gap Statistic", title = "Gap Statistic plot for Vehicle Dataset")

# Identify the optimal number of clusters
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")
cat("Optimal number of clusters based on the gap statistic: ", optimal_k,"\n")

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

# Create a data frame with the silhouette widths for each value of K
df <- data.frame(k = k.min:k.max, silhouette = silhouette_avg)

# Plot the silhouette widths using ggplot2
ggplot(df, aes(x = k, y = silhouette)) +
  geom_point() +
  geom_line() +
  labs(x = "Number of clusters", y = "Silhouette")

# Find the index of the maximum silhouette width
best_k <- which.max(silhouette_avg) + k.min - 1

# Print the best number of clusters based on the silhouette method
cat("Best number of clusters based on the silhouette method: ", best_k, "\n")

#--------------------------------------------------------------------------------------------------------------

#number of clusters using elbow method
num_clusters <- elbow

# Perform k-means clustering using the most favoured number of clusters in scaled_vehicles_data
if(num_clusters > 1) {
  set.seed(123)
  kmeans_result <- kmeans(scaled_vehicles_data, centers = matrix(rnorm(num_clusters * ncol(scaled_vehicles_data)), ncol = ncol(scaled_vehicles_data)), nstart = 25)

  # Print the kmeans output in scaled_vehicles_data
  cat("kmeans output:\n")
  print(kmeans_result)

  # Calculate BSS, TSS, and WSS indices
  centroid_values <- as.data.frame(kmeans_result$centers)
  colnames(centroid_values) <- colnames(scaled_vehicles_data)
  BSS <- sum(colSums((centroid_values - colMeans(scaled_vehicles_data))^2)) * nrow(scaled_vehicles_data)
  TSS <- sum(apply(scaled_vehicles_data, 2, function(x) sum((x - mean(x))^2)))
  WSS <- sum(kmeans_result$withinss)

  cat("BSS/TSS ratio:", BSS/TSS, "\n")
  cat("TSS_indices : ",TSS, "\n")
  cat("BSS_indices : ",BSS, "\n")
  cat("WSS_indices : ",WSS, "\n")

  # Plot the clustering results
  par(mar=c(1,1,1,1))
  plot(scaled_vehicles_data, col = kmeans_result$cluster)
  points(kmeans_result$centers, col = 1:num_clusters, pch = 8, cex = 2)

  # Calculate silhouette coefficients and plot the silhouette plot
  silhouette_obj <- silhouette(kmeans_result$cluster, dist(scaled_vehicles_data))
  plot(silhouette_obj)

  # Calculate the average silhouette width score
  avg_sil_width <- mean(silhouette_obj[, 3])
  cat("Average Silhouette Width Score:", avg_sil_width, "\n")
} else {
  cat("Cannot perform k-means clustering with only one cluster.\n")
}

#------------------------------------------------------------------------------------------------------------------------------
#part 2

# Perform PCA analysis
vehicle_class <- scaled_vehicles_data[,1]
vehicles <- scaled_vehicles_data[,-1]
vehicles <- scaled_vehicles_data[, -which(names(scaled_vehicles_data) == "Class")]

# Initialize graphics device
dev.new()

# Run PCA
pca_result <- PCA(scaled_vehicles_data, graph=TRUE)

# Create a new dataset with principal components as attributes
num_pcs <- sum(pca_result$eig[2,] <= 0.92)
transformed_vehicles_data <- pca_result$ind$coord[,1:num_pcs]

#--------------------------------------------------------------------------------------------------------------
# Use NbClust to determine the number of clusters in transformed_vehicles_data
set.seed(123)
par(mar=c(1,1,1,1))
# Determine the optimal number of clusters using NbClust
nb_clusters_transf <- NbClust(transformed_vehicles_data, min.nc = 2, max.nc = 10, method = "kmeans",index="all")
# Create a data frame with the clustering indices and the number of clusters
df <- data.frame(Clusters = 2:10, nb_clusters_transf$All.index)

# Melt the data frame to long format
df_long <- reshape2::melt(df, id.vars = "Clusters", variable.name = "Index", value.name = "Value")

# Plot the bar plot using ggplot2
ggplot(df_long, aes(x = Clusters, y = Value, fill = Index)) +
  geom_bar(stat = "identity", position = "dodge") +
  xlab("Number of clusters") +
  ylab("Clustering index") +
  ggtitle("NbClust plot") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  geom_vline(xintercept = nb_clusters_transf$Best.nc[1], linetype = "dashed", color = "blue")

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
gap_stat <- clusGap(transformed_vehicles_data, kmeans, nstart = 25, K.max = 3, B = 50)

# Plot the gap statistic
plot(gap_stat, main = "Gap statistic for k-means clustering")

# Identify the optimal number of clusters
optimal_k <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")
cat("Optimal number of clusters based on the gap statistic: ", optimal_k, "\n")

#--------------------------------------------------------------------------------------------------------------
# Calculate the average silhouette width for different values of k
# Set the range of K values to test

# Loop through each value of K and perform clustering using K-means algorithm
k.min <- 2
k.max <- 5
silhouette_vals <- vector("list", length = k.max - k.min + 1)

for (k in k.min:k.max) {
  km <- kmeans(transformed_vehicles_data, centers = k, nstart = 10)

  # Calculate the silhouette width for each data point
  silhouette_vals[[k - k.min + 1]] <- silhouette(km$cluster, dist(transformed_vehicles_data))
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
# Calculate the between_cluster_sums_of_squares (BSS) and the total_sum_of_squares (TSS)

# Perform k-means clustering on the transformed data with the best k value
set.seed(123)
kmeans_transf <- kmeans(transformed_vehicles_data, elbow)

# Plot the clustering results using elbow method
plot(transformed_vehicles_data, col = kmeans_transf$cluster)
points(kmeans_transf$centers, col = 1:elbow, pch = 8, cex = 2)

# Print the k-means output
print(kmeans_transf)

# Calculate BSS and WSS for transformed_vehicles_data
BSS_transf <- sum(kmeans_transf$size * apply((kmeans_transf$centers - apply(transformed_vehicles_data, 2, mean))^2, 1, sum))
TSS_transf <- sum(apply(transformed_vehicles_data^2, 1, sum)) - (sum(transformed_vehicles_data)^2)/length(transformed_vehicles_data)
WSS_transf <- kmeans_transf$tot.withinss

# Print BSS/TSS ratio, BSS, and WSS for transformed_vehicles_data
cat("BSS/TSS ratio for transformed_vehicles_data: ", BSS_transf/TSS_transf, "\n")
cat("TSS for transformed_vehicles_data: ", TSS_transf, "\n")
cat("BSS for transformed_vehicles_data: ", BSS_transf, "\n")
cat("WSS for transformed_vehicles_data: ", WSS_transf, "\n")

# Print k-means output for transformed_vehicles_data
print(kmeans_transf)

# Plot the clustering results
plot(transformed_vehicles_data, col = kmeans_transf$cluster)
points(kmeans_transf$centers, col = 1:elbow, pch = 8, cex = 2)

# Calculate silhouette coefficients and plot the silhouette plot
silhouette_obj <- silhouette(kmeans_transf$cluster, dist(transformed_vehicles_data))
plot(silhouette_obj)

# Calculating Average silhouette width score
silhouette_avg <- mean(silhouette_obj[, 3])
cat("Average silhouette width score: ", silhouette_avg, "\n")

# Compute Calinski-Harabasz index
ch_index <- cluster.stats(transformed_vehicles_data, kmeans_transf$cluster)[[1]]
# Plot the CH index
ch_data <- data.frame(K = 1:length(ch_index), CH_Index = ch_index)

ggplot(ch_data, aes(x = K, y = CH_Index)) +
  geom_point(color = "blue") +
  labs(x = "Number of Clusters (K)", y = "Calinski-Harabasz Index") +
  ggtitle("Calinski-Harabasz Index for K-means Clustering Results")

# Print the CH index
cat("Calinski-Harabasz Index: ", ch_index, "\n")