# prepare necessary libraries
if(!require(optparse)){
  install.packages("optparse")
  library(optparse)
}
if(!require(smacof)){
  install.packages("smacof")
  library(smacof)
}

option_list = list(make_option(c("-d", "--distance_file"), type = "character", default = NULL, help = "path to distance file"),
                   make_option(c("-i", "--items_file"), type = "character", default = NULL, help = "path to item file"),
                   make_option(c("-o", "--output_folder"), type = "character", default = NULL, help = "path to output folder"),
                   make_option(c("-k", "--dims"), type = "integer", default = 20, help = "largest number of dimensions to look at"),
                   make_option(c("-n", "--n_init"), type = "integer", default = 64, help = "number of random initializations for Kruskal algorithm"),
                   make_option(c("-m", "--max_iter"), type = "integer", default = 1000, help = "maximum number of iterations for Kruskal algorithm"),
                   make_option(c("-s", "--seed"), type = "integer", default = NULL, help = "seed for the random number generator"),
                   make_option(c("-p", "--plot"), action = "store_true", default = FALSE, help = "plot the stress value"),
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
stress_by_dimension = list()

if (opt$metric) {
  # METRIC MDS
  
  if (opt$smacof) {
    # use SMACOF algorithm
    
    # iterate over number of dimensions
    for (num_dims in 1:opt$dims) {
      # local variables for keeping best configuration
      best_stress = 1000 
      points = NULL
      # run metric SMACOF multiple times, keep best result
      for (i in 1:opt$n_init) {
        # new random configuration at each step
        initial_config = matrix(rnorm(num_stimuli*num_dims), ncol = num_dims)
        # run metric SMACOF algorithm
        mds_result = smacofSym(dissimilarities, ndim = num_dims, init = initial_config, verbose=FALSE, itmax = opt$max_iter, type = "ratio")
        # if better than before: store result
        if (mds_result$stress < best_stress) {
          best_stress = mds_result$stress
          points = mds_result$conf
        }
      }
      
      # save vectors in file
      output = cbind(item_names, points)
      output_file_name = paste0(opt$output_folder, num_dims, "D-vectors.csv")
      write.table(output, output_file_name, sep = ',', row.names = FALSE, col.names = FALSE, quote = FALSE)
      
      # print stress and store it for plotting
      print(paste(num_dims, best_stress, sep=','), quote = FALSE)
      stress_by_dimension = c(stress_by_dimension, best_stress)
    }
    
    plot_title = "Stress of metric MDS (SMACOF)"

  } else {
    # use classical MDS
    
    # iterate over number of dimensions
    for (num_dims in 1:opt$dims) {
      # Perform Metric MDS
      metric_mds_result = cmdscale(dissimilarities, k = num_dims)
      
      # write vectors to file
      metric_output = cbind(item_names, metric_mds_result)
      metric_output_file_name = paste0(opt$output_folder, num_dims, "D-vectors.csv")
      write.table(metric_output, metric_output_file_name, sep = ',', row.names = FALSE, col.names = FALSE, quote = FALSE)
      
      stress = stress0(dissimilarities, metric_mds_result, type = "ordinal")
      
      # print stress, store it for later plotting
      print(paste(num_dims, stress, sep=','), quote = FALSE)
      stress_by_dimension = c(stress_by_dimension, stress)
    }
    plot_title = "Stress of metric MDS (classical)"
  }
} else {
  # NONMETRIC MDS
  
  if (opt$smacof) {
    # use SMACOF algorithm

    # iterate over number of dimensions
    for (num_dims in 1:opt$dims) {
      # local variables for keeping best configuration
      best_stress = 1000 
      points = NULL
      # run metric SMACOF multiple times, keep best result
      for (i in 1:opt$n_init) {
        # new random configuration at each step
        initial_config = matrix(rnorm(num_stimuli*num_dims), ncol = num_dims)
        # run metric SMACOF algorithm
        mds_result = smacofSym(dissimilarities, ndim = num_dims, init = initial_config, verbose=FALSE, itmax = opt$max_iter, type = "ordinal")
        # if better than before: store result
        if (mds_result$stress < best_stress) {
          best_stress = mds_result$stress
          points = mds_result$conf
        }
      }
      
      # save vectors in file
      output = cbind(item_names, points)
      output_file_name = paste0(opt$output_folder, num_dims, "D-vectors.csv")
      write.table(output, output_file_name, sep = ',', row.names = FALSE, col.names = FALSE, quote = FALSE)
      
      # print stress and store it for plotting
      print(paste(num_dims, best_stress, sep=','), quote = FALSE)
      stress_by_dimension = c(stress_by_dimension, best_stress)
    }
    
    plot_title = "Stress of nonmetric MDS (SMACOF)"
  } else {
    # use Kruskal's algorithm
    
    if(!require(MASS)){
      install.packages("MASS")
      library(MASS)
    }
    # iterate over number of dimensions
    for (num_dims in 1:opt$dims) {
      # local variables for keeping best configuration
      best_stress = 1000 
      points = NULL
      # run nonmetric MDS multiple times
      for (i in 1:opt$n_init) {
        # new random configuration at each step
        initial_config = matrix(rnorm(num_stimuli*num_dims), ncol = num_dims)
        # run Kruskal's algorithm
        kruskal_mds_result = isoMDS(dissimilarities, k = num_dims, y = initial_config, trace=FALSE, maxit = opt$max_iter)
        # if better than before: store result
        current_stress = kruskal_mds_result$stress / 100
        if (current_stress < best_stress) {
          best_stress = current_stress
          points = kruskal_mds_result$points
        }
      }
      
      # save vectors in file
      nonmetric_output = cbind(item_names, points)
      nonmetric_output_file_name = paste0(opt$output_folder, num_dims, "D-vectors.csv")
      write.table(nonmetric_output, nonmetric_output_file_name, sep = ',', row.names = FALSE, col.names = FALSE, quote = FALSE)
      
      # print stress and store it for plotting
      print(paste(num_dims, best_stress, sep=','), quote = FALSE)
      stress_by_dimension = c(stress_by_dimension, best_stress)
    }
    
    plot_title = "Stress of nonmetric MDS (Kruskal)"
    
  }
}

if (opt$plot) {
  # create line plot of stress
  plot_file_name = paste0(opt$output_folder, "Scree.png")
  png(plot_file_name, width = 800, height = 600, unit = "px")
  plot(1:opt$dims, stress_by_dimension, xlab = 'number of dimensions', ylab = 'Stress', main = plot_title)
  lines(1:opt$dims, stress_by_dimension, type='l')
  dev.off()
}
