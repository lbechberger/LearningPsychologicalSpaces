# prepare necessary libraries

# for command line arguments
if(!require(optparse)){
  install.packages("optparse", repos = "http://cran.us.r-project.org")
  library(optparse)
}

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
print("Correlation between features of same rating type")
print("------------------------------------------------")

# preattentive
print("preattentive")
print("  FORM-LINES")
cor.test(data.preattentive$FORM, data.preattentive$LINES, method="pearson")
print("  FORM-ORIENTATION")
cor.test(data.preattentive$FORM, data.preattentive$ORIENTATION, method="pearson")
print("  LINES-ORIENTATION")
cor.test(data.preattentive$LINES, data.preattentive$ORIENTATION, method="pearson")
print("")

#attentive
print("attentive")
print("  FORM-LINES")
cor.test(data.attentive$FORM, data.attentive$LINES, method="pearson")
print("  FORM-ORIENTATION")
cor.test(data.attentive$FORM, data.attentive$ORIENTATION, method="pearson")
print("  LINES-ORIENTATION")
cor.test(data.attentive$LINES, data.attentive$ORIENTATION, method="pearson")
print("")

# comparing attentive and pre-attentive ratings
# ---------------------------------------------
print("Comparison of attentive and pre-attentive ratings")
print("-------------------------------------------------")

# FORM
print("FORM")
t.test(data.preattentive$FORM, data.attentive$FORM, paired = TRUE)
cor.test(data.preattentive$FORM, data.attentive$FORM, method="pearson")

# LINES
print("LINES")
t.test(data.preattentive$LINES, data.attentive$LINES, paired = TRUE)
cor.test(data.preattentive$LINES, data.attentive$LINES, method="pearson")

# ORIENTATION
print("ORIENTATION")
t.test(data.preattentive$ORIENTATION, data.attentive$ORIENTATION, paired = TRUE)
cor.test(data.preattentive$ORIENTATION, data.attentive$ORIENTATION, method="pearson")
