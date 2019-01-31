data("swiss")
head(swiss)

# Load dissimilarity matrix
dissimilarities = read.csv("/home/lbechberger/Documents/GitHub/LearningPsychologicalSpaces/data/NOUN/raw_data/NOUN_distance_matrix.csv", header = FALSE)
dissimilarities = as_tibble(dissimilarities)
head(dissimilarities)

item_names = read.csv("/home/lbechberger/Documents/GitHub/LearningPsychologicalSpaces/data/NOUN/raw_data/item_names.csv", header = FALSE)
head(item_names)


metric_mds_result = cmdscale(dissimilarities)
head(metric_mds_result)

kruskal_mds_result = isoMDS(dissimilarities)
head(kruskal_mds_result)

sammon_mds_result = sammon(dissimilarities)
head(sammon_mds_result)

# CLASSICAL/METRIC MDS

# Load required packages
install.packages("magrittr")
install.packages("dplyr")
install.packages("ggpubr")
library(magrittr)
library(dplyr)
library(ggpubr)
# Cmpute MDS
mds <- swiss %>%
  dist() %>%     # compute distances     
  cmdscale() %>% # run metric MDS
  as_tibble()    # convert to data frame
colnames(mds) <- c("Dim.1", "Dim.2")
# Plot MDS
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(swiss),
          size = 1,
          repel = TRUE)

# NONMETRIC MDS
# --> Kruskal
# Cmpute MDS
library(MASS)
mds <- swiss %>%
  dist() %>%          
  isoMDS() %>%
  .$points %>%
  as_tibble()
colnames(mds) <- c("Dim.1", "Dim.2")
# Plot MDS
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(swiss),
          size = 1,
          repel = TRUE)

# --> Sammon
# Cmpute MDS
library(MASS)
mds <- swiss %>%
  dist() %>%          
  sammon() %>%
  .$points %>%
  as_tibble()
colnames(mds) <- c("Dim.1", "Dim.2")
head(mds)
# Plot MDS
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(swiss),
          size = 1,
          repel = TRUE)

# vegan
install.packages("vegan")
library(vegan)
example_NMDS=metaMDS(swiss, # Our community-by-species matrix
                     k=2) # The number of reduced dimensions
stressplot(example_NMDS)
plot(example_NMDS)
head(example_NMDS)
#ggscatter doesn't work, yet
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(swiss),
          size = 1,
          repel = TRUE)
