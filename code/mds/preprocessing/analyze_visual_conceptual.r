# prepare necessary libraries

# for command line arguments
if(!require(optparse)){
  install.packages("optparse", repos = "http://cran.us.r-project.org")
  library(optparse)
}
# for clmm models
if(!require(ordinal)){
  install.packages("ordinal", repos = "http://cran.us.r-project.org")
  library(ordinal)
}
# for ANOVA
if(!require(car)){
  install.packages("car", repos = "http://cran.us.r-project.org")
  library(car)
}
# for ANOVA
if(!require(FactoMineR)){
  install.packages("FactoMineR", repos = "http://cran.us.r-project.org")
  library(FactoMineR)
}
# for ANOVA
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager", repos = "http://cran.us.r-project.org")
if(!require(mixOmics)){
  BiocManager::install("mixOmics")
}
# for ANOVA
if(!require(RVAideMemoire)){
  install.packages("RVAideMemoire", repos = "http://cran.us.r-project.org")
  library(RVAideMemoire)
}
# for t test
if(!require(pastecs)){
  install.packages("pastecs", repos = "http://cran.us.r-project.org")
  library(pastecs)
}
# for plots
if(!require(ggplot2)){
  install.packages("ggplot2", repos = "http://cran.us.r-project.org")
  library(ggplot2)
}

# parse the command line arguments
option_list = list(make_option(c("-i", "--input_file"), type = "character", default = NULL, help = "path to input file"))

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$input_file)) {
  print_help(opt_parser)
  stop("Missing non-optional argument!", call. = FALSE)
}

# read in the data and ensure correct data types
data <- read.csv(opt$input_file)
data$ratings <- factor(data$ratings, ordered=T)
data$pairID <-as.factor(data$pairID)


# check the coding
str(data)

# TODO: helper function for clmm on conceptual vs visual

# TODO: call helper function on all data, within pairs, between pairs

# TODO: spearman correlation of visual vs conceptual ratings