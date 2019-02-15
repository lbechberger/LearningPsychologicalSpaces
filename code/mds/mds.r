# prepare necessary libraries
if(!require(optparse)){
  install.packages("optparse")
  library(optparse)
}
if(!require(smacof)){
  install.packages("smacof")
  library(smacof)
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
                   make_option(c("-m", "--max_iter"), type = "integer", default = 1000, help = "maximum number of iterations for Kruskal algorithm"),
                   make_option(c("-s", "--seed"), type = "integer", default = NULL, help = "seed for the random number generator"),
                   make_option(c("--metric"), action = "store_true", default = FALSE, help = "use metric instead of nonmetric MDS"),
                   make_option(c("--smacof"), action = "store_true", default = FALSE, help = "use SMACOF algorithm"))

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$distance_file) || is.null(opt$items_file) || is.null(opt$output_folder)) {
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

# collecting the stress for each of the spaces
# dimensions_collection = list()
# metric_stress_collection = list()
# nonmetric_stress_collection = list()

stress_output = NULL

# iterate over number of dimensions
for (num_dims in 1:opt$dims) {
  
  if (opt$metric && !opt$smacof) {
    # classical MDS: only need to run once
    points = cmdscale(dissimilarities, k = num_dims)
    
  } else {
    # all other variants: start with multiple random configurations, keep best
    
    # local variables for keeping best configuration
    best_stress = 1000 
    points = NULL
    
    for (i in 1:opt$n_init) {
      # new random configuration at each step
      initial_config = matrix(rnorm(num_stimuli*num_dims), ncol = num_dims)
      
      # run MDS algorithm and keep current_points and current_stress
      if (opt$metric) {
        # metric SMACOF
        mds_result = smacofSym(dissimilarities, ndim = num_dims, init = initial_config, verbose=FALSE, itmax = opt$max_iter, type = "ratio")
        current_points = mds_result$conf
        current_stress = mds_result$stress
      } else {
        if (opt$smacof) {
          # nonmetric SMACOF
          mds_result = smacofSym(dissimilarities, ndim = num_dims, init = initial_config, verbose=FALSE, itmax = opt$max_iter, type = "ordinal")
          current_points = mds_result$conf
          current_stress = mds_result$stress
        } else {
          # Kruskal's nonmetric MDS
          mds_result = isoMDS(dissimilarities, k = num_dims, y = initial_config, trace=FALSE, maxit = opt$max_iter)
          # if better than before: store result
          current_points = mds_result$points
          current_stress = mds_result$stress / 100
        }
      }
      
      # if better than before: store result
      if (current_stress < best_stress) {
        best_stress = current_stress
        points = current_points
      }
    }
    
  }
  
  # compute metric and nonmetric stress
  metric_stress = stress0(dissimilarities, points, type = "ratio")
  nonmetric_stress = stress0(dissimilarities, points, type = "ordinal")
  
  # save vectors in file
  output = cbind(item_names, points)
  output_file_name = paste0(opt$output_folder, num_dims, "D-vectors.csv")
  write.table(output, output_file_name, sep = ',', row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # store stress for later output
  if (is.null(stress_output)) {
    stress_output = data.frame(num_dims, metric_stress, nonmetric_stress)
    names(stress_output) = c('dim', 'metric_stress', 'nonmetric_stress')
  } else {
    stress_output = rbind(stress_output, c(num_dims, metric_stress, nonmetric_stress))
  }
}

# output stress into csv file for later analysis
stress_file_name = paste0(opt$output_folder, "stress.csv")
write.table(stress_output, stress_file_name, sep = ',', row.names = FALSE, col.names = TRUE, quote = FALSE)