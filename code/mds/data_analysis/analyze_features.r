# prepare necessary libraries

# for command line arguments
if(!require(optparse)){
  install.packages("optparse", repos = "http://cran.us.r-project.org")
  library(optparse)
}
## for t test
#if(!require(pastecs)){
#  install.packages("pastecs", repos = "http://cran.us.r-project.org")
#  library(pastecs)
#}


# parse the command line arguments
option_list = list(make_option(c("-f", "--features_file"), type = "character", default = NULL, help = "path to input file with all feature ratings"))

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$features_file)) {
  print_help(opt_parser)
  stop("Missing non-optional argument!", call. = FALSE)
}

# read in the data and ensure correct data types
data.raw <- read.csv2(opt$features_file, header=TRUE, sep = ",")
data.raw$ratings <- as.numeric(data.raw$ratings)

str(data.raw)
