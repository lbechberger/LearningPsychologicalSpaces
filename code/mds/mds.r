# prepare necessary libraries
if(!require(optparse)){
  install.packages("optparse")
  library(optparse)
}
if(!require(MASS)){
  install.packages("MASS")
  library(MASS)
}

option_list = list(make_option(c("-d", "--distance_file"), type = "character", default = NULL, help = "path to distance file"),
                   make_option(c("-i", "--items_file"), type = "character", default = NULL, help = "path to item file"),
                   make_option(c("-o", "--output_folder"), type = "character", default = NULL, help = "path to output folder"),
                   make_option(c("-k", "--dims"), type = "integer", default = 20, help = "largest number of dimensions to look at"),
                   make_option(c("-n", "--n_init"), type = "integer", default = 64, help = "number of random initializations for Kruskal algorithm"),
                   make_option(c("-s", "--seed"), type = "integer", default = NULL, help = "seed for the random number generator"),
                   make_option(c("-p", "--plot"), action = "store_true", help = "plot the stress value"))
opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$distance_file) || is.null(opt$items_file)) {
  print_help(opt_parser)
  stop("Missing non-optional argument!", call. = FALSE)
}
if (!is.null(opt$seed)) {
  set.seed(opt$seed)
}

# Load dissimilarity matrix
dissimilarities = read.csv(opt$distance_file, header = FALSE)
dissimilarities = as.matrix(dissimilarities)
num_stimuli = sqrt(length(dissimilarities))

# Load item names
item_names = read.csv(opt$items_file, header = FALSE)

# Perform Metric MDS
for (num_dims in 1:opt$dims) {
  metric_mds_result = cmdscale(dissimilarities, k = num_dims)
  metric_output = cbind(item_names, metric_mds_result)
  metric_output_file_name = paste0(opt$output_folder, "/metric/", num_dims, "D-vectors.csv")
  write.table(metric_output, metric_output_file_name, sep = ',', row.names = FALSE, col.names = FALSE)
  
  # TODO: compute, print, and store metric stress
}

# Perform Nonmetric MDS
for (num_dims in 1:opt$dims) {
  best_stress = 1000 
  points = NULL
  for (i in 1:opt$n_init) {
    # new random configuration at each step
    initial_config = matrix(rnorm(num_stimuli*num_dims), ncol = num_dims)
    kruskal_mds_result = isoMDS(dissimilarities, k = num_dims, y = initial_config, trace=FALSE, maxit = 1000)
    # if better than before: store result
    if (kruskal_mds_result$stress < best_stress) {
      best_stress = kruskal_mds_result$stress
      points = kruskal_mds_result$points
    }
  }
  nonmetric_output = cbind(item_names, points)
  nonmetric_output_file_name = paste0(opt$output_folder, "/nonmetric/", num_dims, "D-vectors.csv")
  write.table(nonmetric_output, nonmetric_output_file_name, sep = ',', row.names = FALSE, col.names = FALSE)
  
  print(paste(num_dims, best_stress, sep=','))
  # TODO: store nonmetric stress
}

if (opt$plot) {
  # TODO: plot stress
  
}
