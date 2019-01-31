data("swiss")
head(swiss)

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
  dist() %>%          
  cmdscale() %>%
  as_tibble()
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
# Plot MDS
ggscatter(mds, x = "Dim.1", y = "Dim.2", 
          label = rownames(swiss),
          size = 1,
          repel = TRUE)

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

