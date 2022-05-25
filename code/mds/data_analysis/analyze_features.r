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
data.raw$FORM <- as.numeric(data.raw$FORM)
data.raw$LINES <- as.numeric(data.raw$LINES)
data.raw$ORIENTATION <- as.numeric(data.raw$ORIENTATION)


# split up the data according to ratingType (pre-attentive vs. attentive)
data.preattentive <- subset(data.raw, ratingType=="pre-attentive")
data.attentive <- subset(data.raw, ratingType=="attentive")

# correlation between features within one category
# ------------------------------------------------

# preattentive
cor.test(data.preattentive$FORM, data.preattentive$LINES, method="pearson")
cor.test(data.preattentive$FORM, data.preattentive$ORIENTATION, method="pearson")
cor.test(data.preattentive$LINES, data.preattentive$ORIENTATION, method="pearson")

#attentive
cor.test(data.attentive$FORM, data.attentive$LINES, method="pearson")
cor.test(data.attentive$FORM, data.attentive$ORIENTATION, method="pearson")
cor.test(data.attentive$LINES, data.attentive$ORIENTATION, method="pearson")

