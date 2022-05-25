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

# parse the command line arguments
option_list = list(make_option(c("-c", "--conceptual_file"), type = "character", default = NULL, help = "path to input file with conceptual ratings"),
                   make_option(c("-v", "--visual_file"), type = "character", default = NULL, help = "path to input file with visual ratings"),
                   make_option(c("-t", "--test_assumptions"), action = "store_true", default = FALSE, help = "test CLMM assumptions"),
                   make_option(c("--verbose"), action = "store_true", default = FALSE, help = "output all models"))

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$conceptual_file) || is.null(opt$visual_file)) {
  print_help(opt_parser)
  stop("Missing non-optional argument!", call. = FALSE)
}

# read in the data from the two files, concatenate it, and ensure correct data types
data_conceptual <- read.csv2(opt$conceptual_file, header=TRUE, sep = ",")
data_visual <- read.csv2(opt$visual_file, header=TRUE, sep = ",")

data <- rbind(data_conceptual, data_visual)

data$ratings <- factor(data$ratings, ordered=T)
data$pairID <-as.factor(data$pairID)



# ------------------------------------------------------------------------------
# first look at the full dataset and investigate influence of ratingType

print("CLMM analysis on full data")
print("--------------------------")
print("  Fitting alternative hypothesis (main effect of ratingType)")
# build mixed effects model (CLMM): 
#   do ratings depend on rating type? --> ratings ~ ratingType
#   taking into account that individual picture pair may influence rating --> (1|pairID)
#   taking into account that difference between rating types may differ among picture pairs --> (1 + ratingType | pairID)
model.all <<- clmm(ratings ~ ratingType + (1+ratingType|pairID),  data = data)
if(opt$verbose) {
  print(model.all)
  print('')
}


# test model assumptions if user wants us to (computationally very expensive!)
if (opt$test_assumptions) {
  print("  Testing model assumptions with CLM...")
  
  model.all.test <- clm(ratings ~ ratingType + pairID, data = data)
  
  # first: proportional odds/equal slopes
  print(nominal_test(model.all.test))
  print('')
  
  # second: scale effects 
  print(scale_test(model.all.test))
  print('')
}

# null hypothesis: ratingType does not matter
print("  Fitting null hypothesis (no effect of ratingType)")
model.all.null <<- clmm(ratings ~ 1 + (1+ratingType|pairID), data = data, control=clmm.control(gradTol = 3e-4))
if(opt$verbose) {
  print(model.all.null)
  print('')
}

# conducting likelihood ratio test
print("  Likelihood ratio test...")
print(anova(model.all, model.all.null))
print('')




#-------------------------------------------------------------------------------
# look at between-category ratings and see whether the ratingType effect persists

print("CLMM analysis on between-category data")
print("--------------------------------------")
print("  Fitting alternative hypothesis (main effect of ratingType)")
between <- subset(data, pairType=="between")
# same model as above, but only subset of data
model.between <<- clmm(ratings ~ ratingType + (1+ratingType|pairID),  data = between)
if(opt$verbose) {
  print(model.between)
  print('')
}

# test model assumptions if user wants us to (computationally very expensive!)
if (opt$test_assumptions) {
  print("  Testing model assumptions with CLM...")
  
  model.between.test <- clm(ratings ~ ratingType + pairID, data = between)
  
  # first: proportional odds/equal slopes
  print(nominal_test(model.between.test))
  print('')
  
  # second: scale effects 
  print(scale_test(model.between.test))
  print('')
}


print("  Fitting null hypothesis (no effect of ratingType)")
# same null hypothesis as above, but only subset of data
model.between.null <<- clmm(ratings ~ 1 + (1+ratingType|pairID), data=between, control=clmm.control(gradTol = 3e-4))
if(opt$verbose) {
  print(model.between.null)
  print('')
}

# conducting likelihood ratio test
print("  Likelihood ratio test...")
print(anova(model.between, model.between.null))
print('')




#-------------------------------------------------------------------------------
# now look at within-category ratings, investigating effects of both ratingType and visualType
print("CLMM analysis on within-category data")
print("--------------------------------------")
print("  Fitting alternative hypothesis (interaction betwen ratingType and visualType)")
within <- subset(data, pairType=="within")
# essentially same model, but also considering visualType of the category (VC/VV) as predictor
#   "ratingType * visualType" also allows for interaction of the two predictors
model.within <<- clmm(ratings ~ ratingType * visualType + (1+ratingType|pairID),  data = within)
if(opt$verbose) {
  print(model.within)
  print('')
}

# test model assumptions if user wants us to (computationally very expensive!)
if (opt$test_assumptions) {
  print("  Testing model assumptions with CLM...")
  
  model.within.test <- clm(ratings ~ ratingType * visualType + pairID, data = within)
  
  # first: proportional odds/equal slopes
  print(nominal_test(model.within.test))
  print('')
  
  # second: scale effects 
  print(scale_test(model.within.test))
  print('')
}

# first null hypothesis: no interaction of ratingType and visualType
print("  Fitting null hypothesis (no interaction of ratingType and visualType)")
model.within.null.noInteraction <<- clmm(ratings ~ ratingType + visualType + (1+ratingType|pairID), data = within)
if(opt$verbose) {
  print(model.within.null.noInteraction)
  print('')
}

print("  Likelihood ratio test...")
print(anova(model.within, model.within.null.noInteraction))
print('')

# second null hypothesis: no main effect of ratingType
print("  Fitting null hypothesis (no effect of ratingType)")
model.within.null.ratingType <<- clmm(ratings ~ visualType + (1+ratingType|pairID), data=within, control=clmm.control(gradTol = 3e-4))
if(opt$verbose) {
  print(model.within.null.ratingType)
  print('')
}

# conducting likelihood ratio test
print("  Likelihood ratio test...")
print(anova(model.within, model.within.null.ratingType))
print('')

# third null hypothesis: no main effect of visualType
print("  Fitting null hypothesis (no effect of visualType)")
model.within.null.visualType <<- clmm(ratings ~ ratingType + (1+ratingType|pairID), data=within, control=clmm.control(gradTol = 3e-4))
if(opt$verbose) {
  print(model.within.null.visualType)
  print('')
}

# conducting likelihood ratio test
print("  Likelihood ratio test...")
print(anova(model.within, model.within.null.visualType))
print('')
