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
option_list = list(make_option(c("-c", "--conceptual_file"), type = "character", default = NULL, help = "path to input file with conceptual ratings"),
                   make_option(c("-v", "--visual_file"), type = "character", default = NULL, help = "path to input file with visual ratings"))

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

if (is.null(opt$conceptual_file) || is.null(opt$visual_file)) {
  print_help(opt_parser)
  stop("Missing non-optional argument!", call. = FALSE)
}

# read in the data from the two files, concatenate it, and ensure correct data types
data_conceptual <- read.csv2(opt$conceptual_file)
data_visual <- read.csv2(opt$visual_file)

data <- rbind(data_conceptual, data_visual)

data$ratings <- factor(data$ratings, ordered=T)
data$pairID <-as.factor(data$pairID)

# helper function for running clmm on conceptual vs visual on a given data table
run_clmm <- function(input_data) {
  #str(input_data)
  
  # build mixed effects model (CLMM): 
  #   do ratings depend on rating type? --> ratings ~ ratingType
  #   taking into account that individual picture pair may influence rating --> (1|pairID)
  #   taking into account that difference between rating types may differ among picture pairs --> (1 + ratingType | pairID)

  model <<- clmm(ratings ~ ratingType + (1+ratingType|pairID),  data=input_data)
  print(model)
  print('')
  # test model assumptions:
  model.test <- clm(ratings ~ ratingType + pairID, data=input_data)
  print(model.test)
  # 1) (partial) proportional odds/equal slopes (= der Effekt von ratingType muss konstant sein f?r jeden Anstieg im Rating-Wert 1vs.2, 2vs.3, etc)
  print(nominal_test(model.test))
  print('')
  #> annahme ist erf?llt, wenn p>.05 (als kein sign. Ergebnis)
  
  # 2) keine scale effects (= Skalen-Punkte wurden in beiden Studien und f?r alle Bildpaare gleich interpretiert und auch genutzt, also nicht in manchen F?llen z.B. Extrempunkte gemieden, nur mittelster Wert gew?hlt etc.)
  print(scale_test(model.test))
  print('')
  # > Annahme ist erf?llt, wenn p>.05 (also kein sign. Ergebnis)
  
  
#  Anova(model, type="II")
  
  # check significance of ratingType effect
  #   null hypothesis: ratingType does not matter
  model.null <<- clmm(ratings ~ 1 + (1+ratingType|pairID),data=input_data, control=clmm.control(gradTol = 3e-4))
  print(model.null)
  print('')
  
  print(anova(model, model.null))
  print('')
  
  # TODO: check significance of visualType effect and interaction of effects
}

# TODO: call helper function on all data, within pairs, between pairs
run_clmm(data)

between <- subset(data, pairType=="between")
#run_clmm(between)

within <- subset(data, pairType=="within")
#run_clmm(within)

# TODO ratingType and visualType effects

# TODO: spearman correlation of visual vs conceptual ratings