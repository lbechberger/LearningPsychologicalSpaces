# LearningPsychologicalSpaces
- v0.1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1220053.svg)](https://doi.org/10.5281/zenodo.1220053)
- v1.1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3340766.svg)](https://doi.org/10.5281/zenodo.3340766)
- v1.2: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3712917.svg)](https://doi.org/10.5281/zenodo.3712917)
- v1.3: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4061287.svg)](https://doi.org/10.5281/zenodo.4061287)
- v1.4: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4727605.svg)](https://doi.org/10.5281/zenodo.4727605)
- v1.5: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5524374.svg)](https://doi.org/10.5281/zenodo.5524374)

The code in this repository explores learning a mapping from images to psychological similarity spaces with neural networks.
It has been used as a basis for the following publications:
- Lucas Bechberger and Elektra Kypridemou. "Mapping Images to Psychological Similarity Spaces Using Neural Networks". 6th International Workshop on Artificial Intelligence and Cognition, Palermo/Italy, July 2018. [Paper](http://ceur-ws.org/Vol-2418/) [Preprint](https://arxiv.org/abs/1804.07758) [Release v0.1](https://doi.org/10.5281/zenodo.1220053)
- Lucas Bechberger and Kai-Uwe Kühnberger. "Generalizing Psychological Similarity Spaces to Unseen Stimuli - Combining Multidimensional Scaling with Artificial Neural Networks". In Lucas Bechberger, Kai-Uwe Kühnberger, and Mingya Liu: "Concepts in Action: Representation, Learning, and Application" Language, Cognition, and Mind. Springer (forthcoming) [Preprint](https://arxiv.org/abs/1908.09260) [Release v1.3](https://doi.org/10.5281/zenodo.4061287)
- Lucas Bechberger and Kai-Uwe Kühnberger. "Grounding Psychological Shape Space in Convolutional Neural Networks". (in preparation) [Release v1.5](https://doi.org/10.5281/zenodo.5524374)

## Table of Contents

- [1 About](#1-about) 
  - [1.1 The NOUN study](#11-the-noun-study)
  - [1.2 The Shapes study](#12-the-shapes-study)
- [2 Multidimensional Scaling](#2-multidimensional-scaling)
  - [2.1 Preprocessing](#21-preprocessing)
    - [2.1.1 Parsing NOUN Similarity Data](#211-parsing-noun-similarity-data)
    - [2.1.2 Parsing Shapes Similarity Data](#212-parsing-shapes-similarity-data)
    - [2.1.3 Aggregating Similarity Ratings](#213-aggregating-similarity-ratings)
    - [2.1.4 Parsing Shape Features Data](#214-parsing-shape-features-data)
    - [2.1.5 Exporting Feature Data to CSV](#215-exporting-feature-data-to-csv)
    - [2.1.6 Creating Features from Category Information](#216-creating-features-from-category-information)
  - [2.2 Analysis of the Data Set](#22-analysis-of-the-data-set)
    - [2.2.1 Correlations between Psychological Features](#221-correlations-between-psychological-features)
    - [2.2.2 Creating Average Images](#222-creating-average-images)
    - [2.2.3 Comparing Visual and Conceptual Similarity Ratings](#223-comparing-visual-and-conceptual-similarity-ratings)
    - [2.2.4 Creating Visualizations of the Similarity Matrices](#224-creating-visualizations-of-the-similarity-matrices)
  - [2.3 Creating Similarity Spaces](#23-creating-similarity-spaces)
    - [2.3.1 Applying MDS](#231-applying-mds)
    - [2.3.2 Normalizing the Similarity Spaces](#232-normalizing-the-similarity-spaces)
    - [2.3.3 Visualizing the Similarity Spaces](#233-visualizing-the-similarity-spaces)
    - [2.3.4 Creating Random Baseline Spaces](#234-creating-random-baseline-spaces)
  - [2.4 Analyzing Correlations between Distances and Dissimilarities](#24-analyzing-correlations-between-distances-and-dissimilarities)
    - [2.4.1 Pixel Baseline](#241-pixel-baseline)
    - [2.4.2 Visualizing the Correlations of the Pixel Baseline](#242-visualizing-the-correlations-of-the-pixel-baseline)
    - [2.4.3 ANN Baseline](#243-ann-baseline)
    - [2.4.4 Feature Baseline](#244-feature-baseline)
    - [2.4.5 Distances in MDS Space](#245-distances-in-mds-space)
    - [2.4.6 Creating Shepard Diagrams](#246-creating-shepard-diagrams)
  - [2.5 Analyzing Conceptual Regions](#25-analyzing-conceptual-regions)
    - [2.5.1 Checking for Overlap](#251-checking-for-overlap)
    - [2.5.2 Analyzing Concept Size](#252-analyzing-concept-size)
  - [2.6 Analyzing Interpretable Directions](#26-analyzing-interpretable-directions)
    - [2.6.1 Finding Interpretable Directions](#261-finding-interpretable-directions)
    - [2.6.2 Comparing Interpretable Directions](#262-comparing-interpretable-directions)
    - [2.6.3 Filtering Interpretable Directions](#263-filtering-interpretable-directions)
    - [2.6.4 Aggregating Evaluation Results for Interpretable Directions](#264-aggregating-evaluation-results-for-interpretable-directions)
- [3 Machine Learning](#3-machine-learning)
  - [3.1 Preparing the Data Set for Machine Learning](#31-preparing-the-data-set-for-machine-learning)
    - [3.1.1 Data Augmentation](#311-data-augmentation)
    - [3.1.2 Visualizing Augmented Images](#312-visualizing-augmented-images)
    - [3.1.3 Defining Regression Targets](#313-defining-regression-targets)
    - [3.1.4 Data Set Creation for Shapes study](#314-data-set-creation-for-shapes-study)
  - [3.2 Linear Regression](#32-linear-regression)
    - [3.2.1 ANN-based Feature Extraction](#321-ann-based-feature-extraction)
    - [3.2.2 Pixel-based Feature Extraction](#322-pixel-based-feature-extraction)
    - [3.2.3 Cluster Analysis of Feature Vectors](#323-cluster-analysis-of-feature-vectors)
    - [3.2.4 Regression and Baselines](#324-regression-and-baselines)
    - [3.2.5 Average Results over Folds](#325-average-results-over-folds)
  - [3.3 Hybrid ANN](#33-hybrid-ann)
    - [3.3.1 Training and Evaluating the ANN](#331-training-and-evaluating-the-ann)
    - [3.3.2 Extracting Bottleneck Layer Activations](#332-extracting-bottleneck-layer-activations)
    - [3.3.3 Average Results over Folds](#333-average-results-over-folds)
    - [3.3.4 Visualizing Reconstructions](#334-visualizing-reconstructions)


## 1 About

Our scripts use TensorFlow 1.10 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project.

The folder `code` contains all python scripts used in our experiments. The usage of these scripts is detailed below.
The folder `data` contains the data used for the NOUN study inside the `NOUN` subfolder. This includes the dissimilarity ratings, the images, as well as all intermediate products created by our scripts. In the subfolder `Shapes` we will at some point add the respective results for the Shape study.

Both studies consist of two parts: The first part focuses on the spaces produced by multidimensional scaling (to be found in the `mds` subfolders of both `code` and `data`), whereas the second part focuses on learning a mapping from images into these similarity spaces (to be found in the `ml` subfolders of both `code` and `data`).

Our training data are the images and similarity ratings of the NOUN database (http://www.sussex.ac.uk/wordlab/noun), which were kindly provided by Jessica Horst and Michael Hout: 
Horst, Jessica S., and Michael C. Hout. "The Novel Object and Unusual Name (NOUN) Database: A collection of novel images for use in experimental research." Behavior research methods 48.4 (2016): 1393-1409.

When executing the python scripts, we recommend to execute them as modules (`python -m path.to.module`) - this is necessary for some scripts that make use of utility functions stored in the module `code/util.py`.

### 1.1 The NOUN study

For the study on the NOUN data set, we compared the spaces obtainable by four MDS algorithms (classical MDS, Kruskal's MDS algorithm, metric SMACOF, and nonmetric SMACOF). We investigated both metric and nonmetric stress. Moreover, we computed the correlations between the pairwise distances in the MDS spaces and the dissimilarity ratings. For the latter analysis, we also compared to a pixel-based and an ANN-based baseline.

In a second part of the study, we then trained regressors from either downsampled images or ANN activation to the similarity spaces, investigating a linear regression, a random forest regression, and a lasso regression. We compared the results obtainable on spaces of different sizes and spaces generated by different MDS algorithms.

The script `code/shell_scripts/pipeline_NOUN.py` automatically executes all scripts necessary to reproduce our results. It requires one argument which can either be `paper` (in order to reproduce the results from our paper "Generalizing Psychological Similarity Spaces to Unseen Stimuli - Combining Multidimensional Scaling with Artificial Neural Networks") or `dissertation` (in order to reproduce a more comprehensive set of results discussed in the dissertation). The script then sets up some shell variables accordingly and calls the following five shell scripts which make up the five processing steps in our setup:
- `code/shell_scripts/NOUN/mds.sh`: Preprocesses the dissimilarity ratings (stored in `data/NOUN/mds/similarities/`), applies the different MDS algorithms for target spaces of different sizes (resulting vectors are stored in `data/NOUN/mds/vectors`), and visualizes the spaces with 2D plots (stored in `data/NOUN/mds/visualizations/spaces/`)
- `code/shell_scripts/NOUN/correlation.sh`: Computes different correlation metrics between the pairwise distances between items in the MDS spaces and the original dissimilarity ratings. Also computes the results for a pixel-based and an ANN-based baseline. All of the resulting values are stored in `data/NOUN/mds/correlations/`. Creates some line graphs illustrating how correlation develops based on the dimensionality of the MDS space, and the block size, respectively. These visualizations are stored in `data/NOUN/mds/visualizations/correlations/`.
- `code/shell_scripts/NOUN/ml_setup.sh`: Creates a machine learning data set by appyling data augmentation to the original images, by preparing the target vectors in the similarity space, and by extracting ANN-based and pixel-based features. The resulting data set is stored in multiple files in `data/NOUN/ml/dataset/`.
- `code/shell_scripts/NOUN/experiment_1.sh`: Executes the first machine learning experiment, where we analyze the performance of different regressors (linear regression, random forest regression, lasso regression) on different feature sets (ANN-based vs. pixel-based), using a fixed target space. The results are stored in `data/NOUN/ml/experiment_1/`.
- `code/shell_scripts/NOUN/experiment_2.sh`: Executes the second machine learning experiment, where we analyze the performance on target spaces of the same size that have been created by different MDS algorithms. The results are stored in `data/NOUN/ml/experiment_2/`.
- `code/shell_scripts/NOUN/experiment_3.sh`: Executes the second machine learning experiment, where we analyze the performance on target spaces of different dimensionality that have been created by a single MDS algorithm. The results are stored in `data/NOUN/ml/experiment_3/`.

The only files necessary to run all of these experiments are `data/NOUN/mds/raw_data/raw_distances.csv` (the original dissimilarity matrix from Horst and Hout's NOUN study), `data/NOUN/mds/raw_data/4D-vectors.csv` (the vectors of their four-dimensional similarity space), `data/NOUN/ml/targets.csv` (defining which similarity spaces are included as possible targets in the machine learning data set), and `data/NOUN/ml/folds.csv` (defining the structure of the folds for the cross validation). If the script `code/shell_scripts/clean_NOUN.sh` is executed, all files and folders except for the ones listed above are deleted.


### 1.2 The Shapes study

Our second study focuses on a single conceptual domain, namely the domain of shapes. The approach taken in this study is an extension of our work on the NOUN data set. It contains both the extraction and analysis of similarity spaces as well as learning a mapping from images into these similarity spaces with convolutional neural networks.

The script `code/shell_scripts/pipeline_Shapes.py` automatically executes all scripts necessary to reproduce our results. It requires one argument which can either be `mds` (in order to reproduce the results from our forthcoming paper analyzing the similarity spaces only), `ml` (in order to reproduce our machine learning results) or `dissertation` (in order to reproduce a more comprehensive set of results presented in the dissertation). The script then sets up some shell variables accordingly and calls the following shell scripts which make up the processing steps in our setup:
- `code/shell_scripts/Shapes/data_analysis.sh`: Preprocesses the input data about conceptual and visual similarity and about three psychological features. Also does some simple analyses of the data set and produces some helpful visualizations.
- `code/shell_scripts/Shapes/space_analysis.sh`: Extracts similarity spaces from the data set and analyzes them with respect to three criteria: Do the distances accurately reflect dissimilarities (compares the MDS spaces to the pixel baseline, the ANN baseline, and a baseline using the psychological features)? Are conceptual regions well-formed (i.e., non-overlapping, small, and convex)? Can the psychological features be identified as directions in the similarity spaces?
- `code/shell_scripts/Shapes/ml_setup.sh`: Creates a machine learning data set by appyling data augmentation to the line drawings as well as the Sketchy and TU Berlin data sets of sketches, by preparing the target vectors in the similarity space, and by extracting ANN-based features. The resulting data set is stored in multiple files in `data/Shapes/ml/dataset/`.
- `code/shell_scripts/Shapes/experiment_1.sh`: Investigate mapping performance of a transfer learning task (linear and lasso regression) on top of the pre-trained photo-based inception-v3 network.
- `code/shell_scripts/Shapes/experiment_2.sh`: Train the modified Sketch-a-Net architecture on the classification task and try to find promising hyperparameter settings through a grid search.
- `code/shell_scripts/Shapes/experiment_3.sh`: Transfer learning (linear and lasso regression) on top of the network configurations from experiment 2.
- `code/shell_scripts/Shapes/experiment_4.sh`: Multi-task learning (network optimizes classification and mapping performance at the same time) for the hyperparameter configurations from experiment 2.
- `code/shell_scripts/Shapes/experiment_5.sh`: Applying the most promising configuration from experiments 1, 3, and 4, respectively, without any further modification to target spaces of different dimensionality.

All files are stored in `data/Shapes` which has the following structure:
- `raw_data`: Contains the original input csv files with all ratings. Must be present to execute our pipeline.
- `images`: Contains the images of our stimuli. Unfortunatley, due to copyright restrictions, we are not allowed to publish the original images online. Our scripts can be run without using the original images, though some results (e.g., the pixel baseline or all ML experiments) can then not be reproduced. Please contact us if you are interested in using the images for your own studies!
  - `Berlin-svg`: Original svg vector graphics of the [TU Berlin data set](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/).
  - `Berlin`: Generated png files of the TU Berlin data set.
  - `Sketchy`: Original png files of the [Sketchy data set](http://sketchy.eye.gatech.edu/).
- `mds`: All the (intermediate) results from our analysis with respect to the psychological ratings.
  - `similarities`: Contains the pre-processed individual and aggregated similarity ratings as well as the vectors produced by MDS, all as pickle files. Distinguishes between visual and conceptual similarity (aggregated by median, subfolder `rating_type`) and between mean and median aggregation (only visual similarity, subfolder `aggregator`).
  - `features`: Contains the pre-processed individual and aggregated ratings with respect to the psychological features as well as two category-based features.
  - `data_set`: Contains the most important information extracted in our analysis in the form of CSV files for easier reuse by other researchers.
    - `individual`: Individual ratings with respect to both `features` and `similarities`.
    - `aggregated`: Aggregated ratings with respect to both `features` and `similarities`.
    - `spaces`: Information about our similarity spaces, containing the `coordinates` of the individual stimuli as well as the `directions` corresponding to the psychological features. In both cases, we make a further distinction into `mean` and `median` aggregation.
  - `analysis`: Contains our analysis results with respect to the similarity spaces: `correlations` to dissimilarities (including the baselines), well-formedness of conceptual `regions`, and the presence of interpretable `directions`. In each case, we make a further distinction into `mean` and `median` aggregation.
  - `visualizations`: Contains various visualizations created by our scripts (`average_images` of the categories, `correlations` between distances and dissimilarities, psychological `features`, the `similarity_matrices`, and of course the similarity `spaces` themselves).
- `ml`: All results from our machine learning experiments.
  - `dataset`: The data set used for training the neural network, structured into five folds. Also contains pickle files with the activation vectors of a pre-trained photo-based CNN for different levels of input noise.
  - `experiment_N`: Results, logs, and network weights for the respective experiment.

As for the NOUN study, the script `code/shell_scripts/clean_Shapes.sh` removes everything but `data/Shapes/raw_data/`, `data/Shapes/images/`, `data/Shapes/ml/folds`, and `data/Shapes/ml/regression_targets.csv`.

## 2 Multidimensional Scaling

The folder `code/mds` contains all scripts necessary for the first part of our studies, where we apply multidimensional scaling to a matrix of dissimilarity ratings and where we analyze the resulting spaces.

### 2.1 Preprocessing

The folder `code/mds/preprocessing` contains various scripts for preprocessing the data set in order to prepare it for multidimensional scaling. Depending on the data set (NOUN vs. Shapes), the first preprocessing step differs (see below), but results in the same output data structure, namely a dictionary with the following elements:
- `'categories'`: A dictionary using category names as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'visSim'`: Is the category visually homogeneous? 'VC' means 'Visually Coherent', 'VV' means 'Visually Variable', and 'x' means unclear. 
  - `'artificial'`: Does the category consist of natural ('nat') or artificial ('art') items? 
  - `'items'`: A list of all items that belong into this category. 
- `'items'`: A dictionary using item names as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'category'`: The name of the category this item belongs to. 
- `'similarities'`: A dictionary using the string representation of sets of two items as keys and dictionaries as values. These dictionaries have the following elements:
  - `'relation'`: Is this a 'within' category or a 'between' category rating? 
  - `'category_type'`: `visSim` information of the category (i.e., 'VV', 'VC', or 'x') or 'Mix' if items from different categories.
  - `'values'`: A list of similarity values (integers from 1 to 5, where 1 means 'no visual similarity' and 5 means 'high visual similarity')
- `'category_names'`: An ordered list of category names, determines the order in which categories shall be iterated over in subsequent scripts.


#### 2.1.1 Parsing NOUN Similarity Data

The script `preprocess_NOUN.py` reads in the distances obtained via SpAM and stores them in a pickle file for further processing. It can be executed as follows from the project's root directory:
```
python -m code.mds.preprocessing.preprocess_NOUN path/to/distance_table.csv path/to/output.pickle
```
The script reads the distance information from the specified csv file and stores it together with some further information in the specified pickle file. 

With respect to the resulting `output.pickle` file, we would like to make the following comments: As the NOUN data set does not divide the stimuli into separate categories, we store all items under a single global category which is considered to consist of artificial stimuli. We assume that the visual homogeneity of this category is unclear. As the individual stimuli do not have meaningful names, we use their IDs also as human readable names.

#### 2.1.2 Parsing Shapes Similarity Data

In order to make the Shapes data processible by our scripts, please run the script `preprocess_Shapes.py` as follows from the project's root directory:
```
python -m code.mds.preprocessing.preprocess_Shape path/to/within.csv path/to/within_between.csv path/to/category_names.csv path/to/item_names.csv path/to/output.pickle path/to/output.csv rating_type
```

The file `within.csv` contains within category similarity judments, the file `within_between.csv` contains similarity ratings both within and between categories. The file `category_names.csv` contains an ordered list of category translations which will be used both to translate the category names and to order the categories as specified in the similarity matrix. The file `item_names.csv` contains a translation of item IDs to item names. The resulting information structure is stored in `output.pickle` according to the format described above. Moreover, using the given `rating_type` the most important information is also exported to `output.csv` which has the header `pairID;pairType;visualType;ratingType;ratings` and one line per rating. This csv file can be imported into R for further statistical analysis of the ratings.

The script takes the following optional parameters:
- `-r` or `--reverse` can be set in order to reverse the order of similarity ratings (necessary when using conceptual similarity, as the scale there is inverted)
- `-s` or `--subset`: Specifies which subset of the similarity ratings to use. Default is `all` (which means that all similarity ratings from both studies are used). Another supported option is `between` where only the ratings from the first study are used. Here, all items that did not appear in the first study are removed from further consideration. A third option is `cats` which only considers the categories used in the second study, but which keeps all items from these categories (also items that were only used in the first, but not in the second study). The fourth option `within` only uses data from the first study.
- `-l` or `--limit`: Limit the number of similarity ratings to use to ensure that an equal amount of ratings is aggregated for all item pairs. 
- `-v` or `--limit_value`: Used to give an explicit value for the limit to use. If not set, the script will use the minimal number of ratings observed for any item pair as limit.
- `--seed`: The given seed is used to initialize the random number generator; if no seed is given, results are not reproducible!

#### 2.1.3 Aggregating Similarity Ratings

The next step in the preprocessing pipeline is to aggregate the individual similarity ratings from the overall data set. This can be done with the script `aggregate_similarities.py`. You can execute it as follows from the project's root directory:
```
python -m code.preprocessing.aggregate_similarities path/to/input_file.pickle path/to/output_file.pickle path/to/output_folder_matrix path/to/output_file.csv rating_type
```

`input_file.pickle` should be the output file generated by any of the two preprocessing scripts, `output_file.pickle` determines where the resulting similarity values are stored. The script also computes a dissimilarity matrix which can be used for MDS and stores it in the given `output_folder_matrix`. Finally, the aggregated similarity information is also stored in CSV format in `output_file.csv`, using the header `pairID,pairType,visualType,ratingType,ratings` and the given `rating_type`. Moreover, some information about the global matrix is printed out.

The script takes the following optional arguments:
- `-m` or `--median`: Use the median instead of the mean for aggregating the similarity ratings across participants.

The file `output_file.pickle` consists of a dictionary with the following content:
- `'categories'`: An ordered list of category names, determines the order in which categories shall be iterated over in subsequent scripts.
- `'items'`: An ordered list of item names of all the items for which the similarity values have been computed. Items are ordered alphabetically within a category and accordings to `'category_names'`.
- `'similarities'`: A quadratic matrix of similarity values. Both rows and columns are ordered like in `'items'`. Values of `nan` are used to indicate that there is no similarity rating available for a pair of stimuli.
- `'dissimilarities'`: A quadratic matrix of dissimilarity values analogous to `'similarities'`. Here, values of 0 indicate missing similarity ratings.
- `'category_similarities'`: A quadratic matrix of category-based similarity ratings (i.e., similarities within and between categories).


#### 2.1.4 Parsing Shape Features Data

The script `preprocess_feature.py` reads in the feature ratings (both pre-attentive and attentive) from two given CSV files, converts and aggregates them, and stores them in a single pickle file for further usage. It furthermore creates individual csv files for the aggregated and individual ratings. It can be invoked as follows:
```
python -m code.mds.preprocessing.preprocess_feature path/to/pre_attentive_ratings.csv path/to/attentive_ratings.csv path/to/category_names.csv path/to/item_names.csv path/to/output.pickle path/to/output_individual.csv path/to/output_aggregated.csv
```
`category_names.csv` and `item_names.csv` should be the same files also used for `preprocess_Shapes.py`. The header of the two output csv files is `item;ratingType;ratings`. The `output.pickle` file contains a dictionary with the following structure:
- `individual`: Stores the individual ratings
  - `pre-attentive`: A dictionary mapping from item names to a list of pre-attentive ratings, all normalized to the range [-1,1].
  - `attentive`: A dictionary mapping from item names to a list of attentive ratings, all normalized to the range [-1,1].
- `aggregated`:
  - `pre-attentive`: A dictionary mapping from item names to a single pre-attentive rating, obtained by computing the average across all individual ratings.
  - `attentive`: A dictionary mapping from item names to a single attentive rating, obtained by computing the average across all individual ratings.
- `classification`:
  - `pre-attentive`: A dictionary mapping with the keys `positive` and `negative` and lists of item names as values. Contains the examples with the highest and lowest aggregated value, respectively.
  - `attentive`: A dictionary mapping with the keys `positive` and `negative` and lists of item names as values. Contains the examples with the highest and lowest aggregated value, respectively.


The script accepts the following optional parameters:
- `-p`or `--plot_folder`: If a plot folder is given, the script creates a scatter plot of attentive vs. pre-attentive ratings and stores it in the given location.
- `-i` or `--image_folder`: Path to the folder containing the images for the items. If given, it will use the item images to create scatter plots. If not given, an ordinary scatter plot will be used.
- `-z` or `--zoom`: Determines the size of the item images in the scatter plot. Defaults to 0.15.
- `-q` or `--quantile`: The quantile to use for determining the set of positive and negative classification examples. Defaults to 0.25 (i.e., top quartile and bottom quartile).


#### 2.1.5 Exporting Feature Data to CSV

In order to allow for an easier analysis in R, one can export all feature ratings into two overall csv files (one for the aggregated and one for the raw ratings) using the script `export_feature_ratings.csv`. Moreover, the script takes the individual similarity ratings and exports them together with the aggregated feature ratings into one global csv file for increased convenience in downstream analysis. The script can be executed as follows:
```
python -m code.mds.preprocessing.export_feature_ratings path/to/feature_folder path/to/similarities.pickle rating_type path/to/output_individual.csv path/to/output_aggregated.csv path/to/output_combined.csv
```
The directory `feature_folder` is searched for pickle files containing feature information and the individual similarity ratings (output of `preprocess_Shapes.py`) are taken from `similarities.pickle`, using the meta-information about the `rating_type` (i.e., conceptual vs visual) given as a separate argument. Individual feature ratings are written to `output_individual.csv` using the header `item,ratingType,feature,ratings` (one row per individual rating). Aggregated feature ratings are written to `output_aggregated.csv` using the header `item,ratingType,feature_1,[...],feature_n` where `feature_i` is replaced by the `i`th feature (one row per item). Finally, the `output_combined.csv` file contains individual similarity ratings (same structure as output by `preprocess_Shapes.py`) along with the aggregated feature ratings for both items in the pair. More specifically, the header looks as follows: `pairID,pairType,visualType,ratingType,ratings,`, followed by colums like `item1_FORM_pre-attentive` for all combinations of item index (1 or 2), feature (depends on the input from `feature_folder`), and feature rating type (attentive or pre-attentive).


#### 2.1.6 Creating Features from Category Information

The script `features_from_categories.py` uses the category structure to create candidate features based on both the `visSim` and the `artificial` information. It loads the raw data from `input.pickle` (output of `preprocess_Shapes.py`) and stores the resulting feature information in the given `output_folder` in two pickle files structured in a way analogous to `preprocess_feature.py`. The script can be executed as follows:
```
python -m code.mds.preprocessing.features_from_categories path/to/input.pickle path/to/output_folder
```


### 2.2 Analysis of the Data Set

The folder `code/mds/data_analysis` contains some scripts for visualizing and analyzing the (preprocessed) data set. While the statistical analyses are done with specialized R scripts, other functionality is provided by python scripts.


#### 2.2.1 Correlations between Psychological Features

The script `compare_features.py` compares the scales of two different psychological features to each other, based on each of the scale types. It can be invoked as follows:
```
python -m code.mds.data_analysis.compare_features path/to/first.pickle path/to/second.pickle path/to/output_folder
```
The data about the first and second feature is read in from `first.pickle` and `second.pickle`, respectively. Both files are the output files of the `preprocess_feature.py` script. Scatter plots of the two features are stored in the given `output_folder`. The script takes the following optional arguments:
- `-i` or `--image_folder`: Path to the folder containing the images for the items. If given, it will use the item images to create scatter plots. If not given, an ordinary scatter plot will be used.
- `-z` or `--zoom`: Determines the size of the item images in the scatter plot. Defaults to 0.15.
- `-f` or `--first_name`: Name to use for the first feature. Defaults to `first`.
- `-s` or `--second_name`: Name to use for the second feature. Defaults to `second`.


#### 2.2.2 Creating Average Images

The script `average_images.py` can be used in order to create an average image for each of the categories. It can be invoked as follows:
```
python -m code.mds.data_analysis.average_images path/to/input_file.pickle path/to/image_folder
```
Here, `input_file.pickle` corresponds to the output file of `preprocess_Shapes.py` or `preprocess_NOUN.py` and `image_folder` points to the folder where all the original images reside. The script takes the following optional arguments:
- `-o` or `--output_folder`: The destination folder for the output images, defaults to `.`, i.e., the current working directory.
- `-r` or `--resolution`: The desired size (width and height) of the output images, defaults to 283 (i.e, the size of the original images from the Shapes data set).
- `-a` or `--aggregator`: The aggregator to use for downscaling the images. Defaults to `mean`. Other possible values are `min`, `max`, and `median`.


#### 2.2.3 Comparing Visual and Conceptual Similarity Ratings

You can use the script `find_item_pair_differences.py` to compare the visual and the conceptual similarity ratings and to find item pairs which have identical or very different ratings:
```
python -m code.mds.data_analysis.find_item_pair_differences path/to/visual.pickle path/to/conceptual.pickle
```
Here, `conceptual.pickle` and `visual.pickle` are the corresponding output files of `aggregate_similarities.py`.


#### 2.2.4 Creating Visualizations of the Similarity Matrices

In order to visualize the similarity matrices, one can use the script `plot_similarity_matrices.py`. This script compares two sets of aggregated similarity ratings, given as `first.pickle` and `second.pickle` (output of `aggregate_similarities.py`). It creates one item-based heatmap (below diagonal: first set of similarities, above diagonal: second set of similarities) and two category-based heatmaps (same structure), and stores them as a two separate images `output_folder/heatmap_First_Second_items.png` and `output_folder/heatmap_First_Second_categories.png`. Moreover, it creates a scatter plot of the values in the two matrices and stores them as `output_folder/scatter_First_Second.png`. The script can be invoked as follows:
```
python -m code.mds.preprocessing.plot_similarity_matrices path/to/first.pickle path/to/second.pickle path/to/output_folder/
```
It takes the following optional arguments:
- `-f` or `--first_name`: Descriptive name for the first set of similarities (defaults to 'First').
- `-s` or `--second_name`: Descriptive name for the second set of similarities (defaults to 'Second').
- `-d` or `--dissimilarities`: Use the dissimilarity matrix (instead of the similarity matrix) for making the scatter plot. Does not affect the heatmaps.


### 2.3 Creating Similarity Spaces

The folder `code/mds/similarity_spaces` contains scripts for transforming the given data set from pairwise similarity ratings into a conceptual space.

#### 2.3.1 Applying MDS

The script `mds.r` runs four different versions of multidimensional scaling based on the implementations in R. More specifically, it uses the Eigenvalue-based classical MDS ([cmdscale](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html)), Kruskal's nonmetric MDS ([isoMDS](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/isoMDS.html)), and both metric and nonmetric SMACOF ([smacofSym](https://cran.r-project.org/web/packages/smacof/smacof.pdf#page.55)). For Kruskal's algorithm and for SMACOF, multiple random starts are used and the best result is kept. You can execute the script as follows from the project's root directory:
```
Rscript code/mds/similarity_spaces/mds.r -d path/to/distance_matrix.csv -i path/to/item_names.csv -o path/to/output/directory
```
Here, `distance_matrix.csv` is a CSV file which contains the matrix of pairwise dissimilarities and `item_names.csv` contains the item names (one name per row, same order as in the distance matrix). These two files have ideally been created by `aggregate_similarities.py`. The resulting vectors are stored in the given output directory. All three of these arguments are mandatory. Moreover, a CSV file is created in the output directory, which stores the stress values (metric stress and three variants of nonmetric stress) for each of the generated spaces. 

In order to specify which MDS algorithm to use, one must set exactly one of the following flags: `--classical`, `--Kruskal`, `--metric_SMACOF`, `--nonmetric_SMACOF`.

The script takes the following optional arguments:
- `-k` or `--dims`: Specifies the maximal number of dimensions to investigate. Default value is 20, which means that the script will run the MDS algorithm 20 times, obtaining spaces of dimensionality 1 to 20.
- `-n` or `--n_init`: Specifies how often the nondeterministic MDS algorithms are restarted with a new random initialization. Of all of these runs, only the best result (i.e., the one with the lowest resulting stress) is kept. Default value here is 64.
- `m` or `--max_iter`: Specifies the maximum number of iterations computed within the nondeterministic MDS algorithms. Default values is 1000.
- `-s` or `--seed`: Specifies a seed for the random number generator in order to make the results deterministic. If no seed is given, then the random number generator is not seeded.
- `-t` or `--tiebreaker`: Specifies the type of tie breaking used in the SMACOF algorithm (possible values: `primary`, `secondary`, `tertiary`, default: `primary`).

We implemented the MDS step in R and not in Python because R offers a greater variety of MDS algorithms. Moreover, nonmetric SMACOF with Python's `sklearn` library produced poor results which might be due to a programming bug.

#### 2.3.2 Normalizing the Similarity Spaces

In order to make the individual MDS solutions more comparable, we normalize them by moving their centroid to the origin and by making sure that their mean squared distance to the origin equals one. This is done by the script `normalize_spaces.py`, which can be invoked by simply giving it the path to the directory containing all the vector files:
```
python -m code.mds.similarity_spaces.normalize_spaces path/to/input_folder path/to/input.pickle path/to/output.pickle
```
The script **overrides** the original CSV files. Moreover, it creates an `output.pickle` based on the normalized vectors and the category structure from `input.pickle` (assumed to be generated by `preprocess_Shapes.py` or `preprocess_NOUN.py`). This `output.pickle` file contains a dictioanry with the following structure:
- `'categories'`: A dictionary using category names as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'visSim'`: Is the category visually homogeneous? 'VC' means 'Visually Coherent', 'VV' means 'Visually Variable', and 'x' means unclear. 
  - `'artificial'`: Does the category consist of natural ('nat') or artificial ('art') items? 
  - `'items'`: A list of all items that belong into this category. 
- `n` (for an integer `n` corresponding to the number of dimensions): Dictionary mapping from item names to vectors in the `n`-dimensional similarity space

The script furthermore accepts the following optional arguments:
- `-b` or `--backup`: Create a backup of the original files. Will be stored in the same folder as the original files, file name is identical, but 'backup' is appended before the file extension.
- `-v` or `--verbose`: Prints some debug information during processing (old centroid, old root mean squared distance to origin, new centroid, new root mean squared distance to origin).

**It is important to run this script before using the MDS spaces for the machine learning task -- only by normalizing the spaces, we can make sure that the MSE values are comparable across spaces!**


#### 2.3.3 Visualizing the Similarity Spaces

The script `visualize_spaces.py` can be used to create two-dimensional plots of the similarity spaces. You can execute it as follows from the project's root directory:
```
python -m code.mds.similarity_spaces.visualize_spaces path/to/vectors.pickle path/to/output_folder/
```
The script reads in the vectors from `vectors.pickle` (output of `normalize_spaces.py`), creates two-dimensional plots for all pairs of dimensions, and stores them in the given `output_folder`. 

The script takes the following optional arguments:
- `-i` or `--image_folder`: Path to a folder where the images of the items are stored. If this is given, then the images from this folder are used in the visualization. If no image folder is given, then data points are labeled with their item ID.
- `-z` or `--zoom`: Determines how much the images are scaled. Default is 0.15.
- `-m` or `--max`: Determines the dimensionality of the largest space to be visualized. Defaults to 10.
- `-d` or `--directions_file`: If a path to a directions file (output of `filter_directions.py`) is given, then the given directions are also included into the plots.
- `-c` or `--criterion`: If directions are plotted, the given criterion decides which ones are used. Defaults to `kappa`. Can also be set to `spearman`.
- `-r` or `--region`: If this flag is set, convex hulls of the different categories are included in the plots.

#### 2.3.4 Creating Random Baseline Spaces

For our later analysis, we will need random configurations of points to serve as a simple baseline. The script `create_baseline_spaces.py` can be used to create such random configurations. It can be invoked as follows:
```
python -m code.mds.similarity_spaces.create_baseline_spaces path/to/individual_ratings.pickle path/to/output.pickle n_spaces max_dims
```
Here, `individual_ratings.pickle` is the output created by `preprocess_NOUN.py` or `preprocess_Shapes.py` and is only used to get a list of all item names. The parameter `n_spaces` gives the number of example spaces to generate for each dimensionality and `max_dims` specifies the maximal number of dimensions to consider. The script takes the following optional arguments:
- `-n` or `--normal`: If this flag is set, normally distributed configurations will be generated.
- `-u` or `--uniform`: If this flag is set, uniformly distributed configurations will be generated.
- `-m` or `--shuffled`: This flag is followed by a list in the form `name_1 path_1 name_2 path_2 ...`, giving paths to vector pickle files (output of `normalize_spaces.py`) and their corresponding human-readable name that are shuffled in order to obtain baseline spaces.
- `-s` or `--seed`: Specifies a seed for the random number generator in order to make the results deterministic. If no seed is given, then the random number generator is not seeded.

Please note that at least one of the distribution types `-u`, `-n`, `-m` must be set. The resulting pickle file contains a hierarchical dictionary mapping using the baseline type, the number of dimensions, and the item as keys on the different hierarchy levels. Please note that for each number of dimensions a list of spaces is stored (of length `n_spaces`).


### 2.4 Analyzing Correlations between Distances and Dissimilarities

The folder `code/mds/correlations` contains various scripts for correlating distances and dissimilarities for the MDS solutions and various baselines. In all cases, we consider three distance measures (Euclidean, Manhattan, inner product) and five correlation metrics (Pearson's r, Spearman's rho, Kendall's tau, and the coefficient of determination R²).

#### 2.4.1 Pixel Baseline

The script `pixel_correlations.py` loads the images and downscales them using scipy's `block_reduce` function (see [here](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce)). Here, all pixels within a block of size `k` times `k` are aggregated via one of the following aggregation functions: maximum, minimum, arithmetic mean, median. The script automatically iterates over all possible combinations of `k` and the aggregation function.

The pixels of the resulting downscaled images are interpreted as one-dimensional feature vectors. All of the distance measures are used to build distance matrices, which are then in turn correlated with the original dissimilarity ratings (both using the raw distances and using a weighted version where dimension weights are estimated in a cross-validation). The script can be executed as follows, where `aggregated_ratings.pickle` is the output file of `aggregate_similarities.py`, `distances_folder` is a folder containing pickle files with precomputed distances, and `output_file.csv` is the destination CSV file which where the results will be stored:
```
python -m code.mds.correlations.pixel_correlations path/to/aggregated_ratings.pickle path/to/distances_folder/ path/to/output_file.csv
```

By default, the pre-computed distances from `distances.pickle` are used to compute the correlations. If however the optional parameter `-i` or `--image_folder` is specified, the images are loaded from that given folder, the distances are computed manually, and are stored in `distances.pickle` for future use.

Please note that one or more of the following flags must be set in order to specify the correlation metric(s) to use in the analysis:
- `--pearson`: Compute Pearson's correlation coefficient (linear correlation).
- `--spearman`: Compute Spearman's correlation coefficient (monotone correlation).
- `--kendall`: Compute Kendall's correlation coefficient (monotone correlation).
- `--r2_linear`: Compute the coefficient of determination R² for a linear regression (linear correlation).
- `--r2_isotonic`: Compute the coefficient of determination R² for an isotonic regression (monotone correlation).

The script takes the following additional optional parameters:
- `-w` or `--width`: The width (and also height) of the full images, i.e., the maximal number of `k` to use (default: 300, i.e. the image size of the NOUN data set).
- `-g` or `--greyscale`: If this flag is set, the three color channels are collapsed into a single greyscale channel when loading the images. If not, full RGB information is used.
- `-n` or `--n_folds`: The number of folds to use in the cross-validation process of optimizing dimension weights (defaults to 5).
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the folds and thus the overall results deterministic. If no seed is given, then the random number generator is not seeded.

**IMPORTANT! For non-greyscale images, this script takes a lot of memory. Use code/shell_scripts/NOUN/pixel_job.sge to run it on the grid instead of locally.**

#### 2.4.2 Visualizing the Correlations of the Pixel Baseline

The script `visualize_pixel_correlations.py` can be used to visualize the results of the pixel baseline as a function of block size. It can be invoked as follows:
```
python -m code.mds.correlations.visualize_correlations path/to/pixel_file.csv path/to/output_folder
```
Here, `pixel_file.csv` is the output file of `pixel_correlations.py` and `output_folder` determines where the resulting plots are stored.

Please note that one or more of the following flags must be set in order to specify the correlation metric(s) to use for visualization:
- `--pearson`: Compute Pearson's correlation coefficient (linear correlation).
- `--spearman`: Compute Spearman's correlation coefficient (monotone correlation).
- `--kendall`: Compute Kendall's correlation coefficient (monotone correlation).
- `--r2_linear`: Compute the coefficient of determination R² for a linear regression (linear correlation).
- `--r2_isotonic`: Compute the coefficient of determination R² for an isotonic regression (monotone correlation).

#### 2.4.3 ANN Baseline

As a second baseline, we use the features extracted by the a neural network (more specifically, the [Inception-v3 network](https://arxiv.org/abs/1512.00567)) to predict the similarities between images from the data set. The corresponding script is called `ann_correlations.py` and is invoked as follows:
```
python -m code.mds.correlations.ann_correlations path/to/model_folder path/to/aggregated_ratings.pickle path/to/distances.pickle path/to/output_file.csv 
```

By default, the pre-computed distances from `distances.pickle` are used to compute the correlations. If however the optional parameter `-i` or `--image_folder` is specified, the images are loaded from that given folder, the distances are computed manually, and are stored in `distances.pickle` for future use. In the latter case, the script downloads the inception network into the given `model_folder`, takes all images from the `image_folder`, and computes the activation of the second-to-last layer of the ANN. This activation vector is then used as a feature vectors. All of the distance measures are used to build distance matrices, which are then in turn correlated with the original dissimilarity ratings from `aggregated_ratings.pickle` (both using the raw distances and using a weighted version where dimension weights are estimated in a cross-validation). 

Please note that one or more of the following flags must be set in order to specify the correlation metric(s) to use in the analysis:
- `--pearson`: Compute Pearson's correlation coefficient (linear correlation).
- `--spearman`: Compute Spearman's correlation coefficient (monotone correlation).
- `--kendall`: Compute Kendall's correlation coefficient (monotone correlation).
- `--r2_linear`: Compute the coefficient of determination R² for a linear regression (linear correlation).
- `--r2_isotonic`: Compute the coefficient of determination R² for an isotonic regression (monotone correlation).

The script takes the following optional arguments:
- `-n` or `--n_folds`: The number of folds to use in the cross-validation process of optimizing dimension weights (defaults to 5).
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the folds and thus the overall results deterministic. If no seed is given, then the random number generator is not seeded.

#### 2.4.4 Feature Baseline

If we interpret the values on the scales of the (psychological) features as coordinates of a similarity space, we can use these coordinates to also compute distances between stimuli. The script `feature_correlations.py` does exactly this and computes the correlation to the original dissimilarity ratings (both using the raw distances and using a weighted version where dimension weights are estimated in a cross-validation). It is called as follows:
```
python -m code.mds.correlations.feature_correlations path/to/aggregated_ratings.pickle path/to/distances.pickle path/to/output.csv
```
Here, `aggregated_ratings.pickle` is again the output file of `aggregate_similarities.py`, `distances.pickle` contains pre-computed distances, and the results will be stored in `output.csv`. By default, the pre-computed distances from `distances.pickle` are used to compute the correlations. If however the optional flag `-f` or `--feature_folder` is given, all pickle files from this given folder are read (assumed to be generated by `preprocess_feature.py` or `features_from_categories.py`) and the underlying aggregated feature ratings are used to compute distances. In the latter case, the resulting distances are stored in `distances.pickle` for future use.


Please note that one or more of the following flags must be set in order to specify the correlation metric(s) to use in the analysis:
- `--pearson`: Compute Pearson's correlation coefficient (linear correlation).
- `--spearman`: Compute Spearman's correlation coefficient (monotone correlation).
- `--kendall`: Compute Kendall's correlation coefficient (monotone correlation).
- `--r2_linear`: Compute the coefficient of determination R² for a linear regression (linear correlation).
- `--r2_isotonic`: Compute the coefficient of determination R² for an isotonic regression (monotone correlation).

The script takes the following optional arguments:
- `-n` or `--n_folds`: The number of folds to use in the cross-validation process of optimizing dimension weights (defaults to 5).
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the folds and thus the overall results deterministic. If no seed is given, then the random number generator is not seeded.

#### 2.4.5 Distances in MDS Space

The script `mds_correlations.py` loads the MDS vectors and derives distances between pairs of stimuli based on the three distance measures. These distances are then correlated to the human dissimilarity ratings (both using the raw distances and using a weighted version where dimension weights are estimated in a cross-validation). The script can be executed as follows:
```
python -m code.mds.correlations.mds_correlations path/to/aggregated_ratings.pickle path/to/distances.pickle path/to/output.csv
```
Again, `aggregated_ratings.pickle` is again the output file of `aggregate_similarities.py`, `distances.pickle` contains pre-computed distances, and the results will be stored in `output.csv`. By default, the pre-computed distances from `distances.pickle` are used to compute the correlations. If however the optional flag `-v` or `--vector_file` is given, then the MDS vectors are loaded from this given pickle file (assumed to be generated by `normalize_spaces.py`) and are used to compute the distances. In the latter case, the resulting distances are stored in `distances.pickle` for future use.

Please note that one or more of the following flags must be set in order to specify the correlation metric(s) to use in the analysis:
- `--pearson`: Compute Pearson's correlation coefficient (linear correlation).
- `--spearman`: Compute Spearman's correlation coefficient (monotone correlation).
- `--kendall`: Compute Kendall's correlation coefficient (monotone correlation).
- `--r2_linear`: Compute the coefficient of determination R² for a linear regression (linear correlation).
- `--r2_isotonic`: Compute the coefficient of determination R² for an isotonic regression (monotone correlation).

The script takes the following optional arguments:
- `-b` or `--baseline_file`: If a file generated by `create_baseline_spaces.py` is given as an argument, the expected correlation value for all of the baseline spaces is also computed and stored.
- `--n_min`: The size of the smallest space to investigate (defaults to 1).
- `--n_max`: The size of the largest space to investigate (defaults to 20).
- `-n` or `--n_folds`: The number of folds to use in the cross-validation process of optimizing dimension weights (defaults to 5).
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the folds and thus the overall results deterministic. If no seed is given, then the random number generator is not seeded.

#### 2.4.6 Creating Shepard Diagrams

For some further visualization, the script `shepard_diagram.py` can be used in order to create a Shepard plot (i.e., a scatter plot of predicted distances versus actual dissimilarities). It is invoked as follows:
```
python -m code.mds.correlations.scatter_plot path/to/aggregated_ratings.pickle path/to/distances.pickle path/to/output_image.png
```
Here, `aggregated_ratings.pickle` is the output of `aggregate_similarities.py`, `distances.pickle` refers to the file generated by the respective correlation script (i.e., `pixel_correlations.py`, `ann_correlations.py`, `feature_correlations.py`, or `mds_correlations.py`), and `output_image.png` is the file name under which the scatter plot will be stored. There are three different modes for the scatter plot generation (based on the three correlation approaches) and exactly one of them must be picked via an optional argument:
- `--mds` or `-m`: The MDS vectors of the given number of dimensions is used.
- `--ann` or `-a`: The ANN baseline is used.
- `--features` or `-f`: The feature baseline with the given feature space is used. The following additional argument needs to be specified for the feature baseline:
	- `--type` or `-t` determines the type of feature ratings to use.
- `--pixel` or `-p`: The pixel baseline with the given aggregator is used. The following additional argument needs to be specified for the pixel baseline:
	- `--block_size` or `-b` determines the block size (defaults to 1).
- `--random` or `-r`: A random configuration of points with the given number of dimensions is used. The following additional argument needs to be specified:
	- `--distribution` determines the underlying probability distribution (defaults to `normal`, options are as generated by `create_baseline_spaces.py`)

For all three of these cases, the parameter `--distance` or `-d` determines which distance function to use (`Euclidean`, `Manhattan`, or `Cosine`).

In its basic version, the script uses fixed identical dimension weights of one. If the flag `--optimized` or `-o` is set, then optimal dimension weights are computed based on a cross-validation approach (identical to the scripts above). The following additional parameters control the behavior of this cross-validation:
- `-n` or `--n_folds`: The number of folds to use in the cross-validation process of optimizing dimension weights (defaults to 5).
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the folds and thus the overall results deterministic. If no seed is given, then the random number generator is not seeded.

Finally, the flag `--similarity_name` can be used to determine the title of the y-axis by specifying which type of similarity file is being used. Its default setting is `'Mean'` (i.e., the y-axis will be labeled as 'Mean Dissimilarity from Psychological Study').


### 2.5 Analyzing Conceptual Regions

The folder `code/mds/regions` contains two scripts for analyzing the well-formedness of conceptual regions. *This is only applicable to the Shapes data set, as there are no categories in NOUN.*

#### 2.5.1 Checking for Overlap

The script `analyze_overlap.py` can be used to check whether the categories within the space are non-overlapping. It iterates over all categories, builds a convex hull of the items belonging to this category and counts how many points from other categories lie within this convex hull. Each point that lies in the convex hull of a different concept is counted as one violation. This analysis takes place both for all categories and for the VC-VV distinction.

The script can be exectued as follows (where `vectors.pickle` is the output of `normalize_vectors.py` and `n_dims` is the dimensionality of the space to consider):
```
python -m code.mds.similarity_spaces.analyze_overlap path/to/vectors.pickle n_dims path/to/output_file.csv
```
The resulting number of violations is stored in `output_file.csv` using the header `dims,hull_category_type,intruder_category_type,data_source,violations`.

The script takes the following optional parameter:
- `-b` or `--baseline`: This argument points to the pickle-file generated by `create_baseline_spaces.py`. If this parameter is given, the script also runs computations for the given baseline spaces.

#### 2.5.2 Analyzing Concept Size

The script `analyze_concept_size.py` evaluates category size by computing the average distance to the category prototype for all categories. 

The script can be invoked as follows (where `vectors.pickle` is the output of `normalize_vectors.py` and `n_dims` is the dimensionality of the space to consider):
```
python -m code.mds.similarity_spaces.analyze_concept_size path/to/vectors.pickle n_dims path/to/output_file.csv
```
The resulting average category sizes are stored in `output_file.csv` using the header `dims,category_type,data_source,size`.

The script takes the following optional parameter:
- `-b` or `--baseline`: This argument points to the pickle-file generated by `create_baseline_spaces.py`. If this parameter is given, the script also runs computations for the given baseline spaces.


### 2.6 Analyzing Interpretable Directions

The folder `code/mds/directions` contains various scripts for extracting interpretable directions in the similarity space based on the given features.

#### 2.6.1 Finding  Interpretable Directions
The script `find_directions.py` tries to find interpretable directions in a given similarity space based on a regression or classification task. *This is only applicable to the Shapes data set, as there are no categories in NOUN.*
It can be invoked as follows (where `n_dims` is the number of dimensions of the underlying space to consider):
```
python -m code.mds.directions.find_directions path/to/vectors.pickle n_dims path/to/feature.pickle path/to/output.csv
```
Here, `vectors.pickle` is the output of `normalize_spaces.py`. Based on the feature information from `feature.pickle` (output of `preprocess_feature.py` or `features_from_categories.py`) the script constructs a classification and a regression problem and trains a linear SVM and a linear regression on them, respectively. The quality of the model fit is evaluated by extracting the normal vector of the separating hyperplane and by projecting all points onto this normal vector. Then, we use Cohen's kappa to measure how well a simple threshold classifier on the resulting values performs. Moreover, we compute the Spearman correlation of the projected vectors to the scale values from the regression problem. The resulting numbers are stored in `output.csv`, along with the extracted direction, using the header `dims,data_source,space_idx,feature_type,model,kappa,spearman,d1,...,d20`.

The script takes the following optional parameter:
- `-b` or `--baseline`: This argument points to the pickle-file generated by `create_baseline_spaces.py`. If this parameter is given, the script also runs computations for the given baseline spaces.


#### 2.6.2 Comparing Interpretable Directions
The script `compare_directions.py` compares the interpretable directions found by `find_directions.py` by using the cosine similarity. More specifically, the script iterates through all spaces with a dimenionality of maximally `n_dims`. For each space, it computes the average cosine similarity of all the interpretable directions for the same feature (which were however constructed based on different feature rating scales and different ML algorithms). Moreover, it computes the average cosine similarity for each pair of features (by comparing all pairs of directions). The results are stored in `output.csv`. The script furthermore requires an `input_folder`, which contains all csv files created by `find_directions.py` (and no additional files!).
The script can be executed as follows:
```
python -m code.mds.directions.compare_directions path/to/input_folder/ n_dims path/to/output.csv
```

#### 2.6.3 Filtering Interpretable Directions
In order to aggregate the different candidate directions for each feature, one can use the script `filter_directions.py`. It can be invoked as follows:
```
python -m code.mds.directions.filter_directions path/to/input_file.csv direction_name n_dims path/to/output.csv
```
The script loads the candidate directions along with their evaluation results from `input_file.csv` (which is the output of `find_directions.py`). For each space (up to `n_dims`), it compares all candidate directions based on Cohen's kappa and based on the Spearman correlation. For each of these evaluation metrics, the directions with the highest values are kept and averaged. The result (the dimensionality of the space, the `direction_name`, the averaged direction and the list of candidate directions it is based on) is written to `output.csv`.
The script takes the following optional parameters:
- `-k` or `--kappa_threshold`: Minimal value of Cohen's kappa required to pass the filter. Defaults to 0.
- `-s` or `--spearman_threshold`: Minimal value of the Spearman correlation required to pass the filter. Defaults to 0.


#### 2.6.4 Aggregating Evaluation Results for Interpretable Directions
In order to make the subsequent analysis easier, the script `aggregate_direction_results.py` can be used to aggregate the evaluation results created by `find_directions.py` based on data_source, direction name, scale type, and ML model (by averaging over the two other conditions). It is executed as follows, where `input_folder` contains the csv files created by `find_directions.py` (and no additional files!), `n_dims` is the maximal dimensionality of the similarity space to consider. The results will be stored as separate csv files inside the given `output_folder`.
```
python -m code.mds.directions.aggregate_direction_results path/to/input_folder/ n_dims path/to/output/folder
```


## 3 Machine Learning

The folder `code/ml` contains all scripts necessary for the second part of our studies, where we apply machine learning techniques in order to learn a mapping from images to points in the similarity space.

### 3.1 Preparing the Data Set for Machine Learning

In order to run a regression from images to MDS coordinates, multiple preprocessing steps are necessary. Firstly, we need to augment our data set by creating a large amount of slightly distorted image variants. This is done in order to achieve a data set of reasonable size for a machine learning task. Moreover, for each of the images, the target MDS coordinates need to be prepared. All scripts for these steps can be found in the `code/ml/preprocessing` folder.

#### 3.1.1 Data Augmentation

We used [ImgAug](https://github.com/aleju/imgaug) for augmenting our image data set for the NOUN data set. This is done with the script `data_augmentation.py`. It can be invoked as follows:
```
python -m code.ml.preprocessing.data_augmentation path/to/image_folder/ path/to/output_folder/ n
```
The script searches for all jpg images in the given `image_folder`, creates `n` augmented samples of each image and stores the results in the given `output_folder` (one pickle file per original image). The script takes the following optional command line arguments:
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the results deterministic. If no seed is given, then the random number generator is not seeded.
- `-i` or `--image_size`: The expected image size in pixels. Defaults to 300 (i.e., the image size of the NOUN data set).

Augmentation is done by appling the folloing operations in random order:
- *Horizontal flips*. The probability of a horizontal flip can be controlled with the optional parameter `--flip_prob` (defaults to 0.5).
- *Cropping*. The maximal relative amount of cropping for each side of the image can be controlled with `--crop_size` (default: 0.1).
- *Gaussian Blur*. The probability of using a Gaussian blur can be set via `--blur_prob` (default: 0.5) and it's sigma value via `--blur_sigma` (default: 0.5)
- *Varying the contrast*. The borders of possible contrast values (relative to the image's original contrast) can be set via `--contrast_min` (default: 0.75) and `--contrast_max` (default: 1.5).
- *Additive Gaussian noise*. The value of sigma is set via `--g_noise_sigma` (default: 0.05) and the probability of drawing a different value for each color channel independently by seeting `--g_noise_channel_prob` (default: 0.5)
- *Varying the brightness*. The borders of possible brightness values (relative to the image's original brightness) can be set via `--light_min` (default: 0.8) and `--light_max` (default: 1.2). Moreover, the probability of drawing a different value for each color channel independently is controlled by `--light_channel_prob` (default: 0.2)
- *Zooming*: The borders of possible zoom values (relative to the image's original size) can be set via `--scale_min` (default: 0.8) and `--scale_max` (default: 1.2).
- *Translation*: The relative amount of translation is set with `--translation` (default: 0.2)
- *Rotation*: The maximal rotation angle in degrees is controlled by `--rotation_angle` (default: 25).
- *Shearing*: The maximal shear angle in degrees is controlled by `--shear_angle` (default: 8).
- *Salt and pepper noise*: The amount of pixels to modify is set via `--sp_noise_prob` (default: 0.03)

#### 3.1.2 Visualizing Augmented Images

In order to visually check that the augmentation step worked for the NOUN data, you can use the script `show_augmented_images.py` to display them. It can be executed as follows:
```
python -m code.ml.preprocessing.show_augmented_images path/to/augmented.pickle
```
Here, `augmented.pickle` is one of the pickle files created by `data_augmentation.py`. By default, the script displays three rows (adjustable via `-r` or `--rows`) and four columns (adjustable via `-c` or `--columns`).

#### 3.1.3 Defining Regression Targets

As our experiments are run against a wide variety of target spaces, we created a script called `prepare_targets.py` which for convenience collects all possible target vectors in a single pickle file. It moreover creates a shuffled version of the targets for later usage as a control case. The script can be invoked as follows:
```
python -m code.ml.preprocessing.prepare_targets path/to/input.csv path/to/output.pickle
```
Here, `input.csv` is a csv file with two columns: In each row, the first column contains a short descriptive name of the target space and the second column contains the path to the corresponding file with the MDS vectors (as created in Section 2.2.1 and normalized in Section 2.2.2). The script iterates through all these target spaces and collects the MDS vectors. When shuffling them, the same seed is used for all spaces to ensure that the results are comparable. By setting `-s` or `--seed`, the user can specify a fixed seed, otherwise a random seed is drawn in the beginning of the script. 

The result is stored in `output.pickle` as a dictionary having the names of the target spaces as keys and further dictionaries (with the keys `correct` and `shuffled` leading to dictionaries with the corresponding image-vector mappings) as values.

#### 3.1.4 Data Set Creation for Shapes study

Since our Shapes study makes use of multiple data sources and a specific augmentation process, we created a separate script called `prepare_Shapes_data.py` for this preprocessing stage. It can be invoked as follows:
```python -m code.ml.preprocessing.prepare_Shapes_data path/to/folds_file.csv path/to/output_directory/ factor```
Here, `folds_file.csv` is a csv file that contains the columns `path` (giving the relative path of the image file from the project's root directory) and `fold` (the fold to which this image belongs). For classification data, a column `class` indicates the image class, while for data with psychological similarity ratings, the column `id` gives the stimulus ID used in the similarity space. The script will read all images listed in the `path` column of the `folds_file.csv`, create `factor` augmented copies of each image (by scaling it to a random size between 168 and 224 and by randomly translating it afterwards). The resulting augmented images will be stored as individual png files in the given `output_directory` and an additional pickle file containing a (shuffled) list of paths and classes/ids is created in the same file.

The script takes the following optional arguments:
- `-p` or `--pickle_output_folder`: If a pickle output similar to the one provided by `data_augmentation.py` is desired, you can define the output folder for the augmented images here.
- `-n` or `--noise_prob`: A list of floats specifying the different noise levels of salt and pepper noise to be added in the pickle versions.
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the results deterministic. If no seed is given, then the random number generator is not seeded.
- `-o` or `--output_size`: Size of the output image, defaults to 224.
- `-m` or `--minimum_size`: Minimal size of the object, defaults to 168.

### 3.2 Linear Regression

As a first pass of the regression task, we evaluate some simple baselines (which disregard the images altogether) as well as some linear regressions based on either downscaled images or the features extracted by a pretrained neural network. All scripts are contained in the `code/ml/regression` folder.

#### 3.2.1 ANN-based Feature Extraction

In order to create feature vectors based on the activations of the [Inception-v3 network](https://arxiv.org/abs/1512.00567), one can use the script `ann_features.py`. It is invoked as follows:
```
python -m code.ml.regression.ann_features path/to/model_folder path/to/input_folder path/to/output.pickle
```
The script downloads the inception-v3 network into the folder specified by `model_folder`, reads all augmented images from the folder specified by `input_folder`, uses them as input to the inception network, grabs the activations of the second-to-last layer of the network (2048 neurons) and stores a dictionary mapping from image name to a list of feature vectors in the pickle file specified by `output.pickle`.

#### 3.2.2 Pixel-based Feature Extraction
In order to create feature vectors by downscaling the original images, one can use the script `pixel_features.py`. It is invoked as follows:
```
python -m code.ml.regression.pixel_features path/to/input_folder path/to/output.pickle
```
The script reads all augmented images from the folder specified by `input_folder`, downscales them them according to the way described already in Section 2.3.1, and stores a dictionary mapping from image name to a list of feature vectors in the pickle file specified by `output.pickle`. It takes the following optional arguments:
- `-a` or `--aggregator`: Type of aggregator function to use. One of `max`, `min`, `mean`, `median` (default: `mean`).
- `-g` or `--greyscale`: If this flag is set, the image is converted to greyscale before downscaling (reduces the number of output features by factor 3).
- `-b` or `--block_size`: Size of one block that will be reduced to a single number. Defaults to 1.

#### 3.2.3 Cluster Analysis of Feature Vectors
The point of data set augmentation is to create a larger variety of input images and to introduce some additional noise into the data set. The script `cluster_analysis.py` takes a file of feature vectors and analyzes whether they form strong clusters (in the sense that all augmented images based on the same original are very similar to each other, but very different from other images). It uses the Silhouette coefficient to quantify this. As comparison, the Silhouette coefficient of a shuffled data set is computed. The script can be called as follows:
```
python -m code.ml.regression.cluster_analysis path/to/features.pickle
```
Here, `features.pickle` is the pickle file generated by either `ann_features.py` or `pixel_features.py`. The script takes the following optional arguments:
- `-n` or `--n_sample`: The number of samples to randomly draw for each original image (defaults to 100). Computing the Silhouette coefficient may be untractable for large data sets, so sampling might be required.
- `-s` or `--seed`: The random seed to use for initializing the random number generator. If none is given, a different initialization is used in every call to the script.

#### 3.2.4 Regression and Baselines

The script `regression.py` can be used to run a linear regression, a lasso regression, or any of the baselines. It is called as follows:
```
python -m code.ml.regression.regression path/to/target_vectors.pickle space_name path/to/features.pickle path/to/folds.csv path/to/output.csv 
```
Here, `target_vectors.pickle` is the file generated by `prepare_targets.py`, `space_name` is the name of a target space contained in this file, `features.pickle` contains the features to be used (either generated by `ann_features.py` or by `pixel_features.py`), `folds.csv` contains the fold structure (for each original image the number of the fold it belongs to), and `output.csv` is the file in which the results will be stored (the script appends to the file if it already exists).

In order to select the type of regression to be used, one needs to pass *exactly one* of the following flags to the script:
- `--zero`: *Zero baseline*, always predicts the origin of the feature space (i.e., a vector where all entries are zero)
- `--mean`: *Mean baseline*, always predicts the mean of the target vectors seen during training.
- `--normal`: *Normal distribution baseline*, estimates a multivariate normal distribution on the target vectors seen during training. Draws a random sample from this distribution for making predictions.
- `--draw`: *Random draw baseline*, uses randomly selected target vectors from the training set for making predictions.
- `--linear`: *Linear regression*, runs sklearn's `LinearRegression` after normalizing the feature space.
- `--lasso`: *Lasso regression, runs sklearn's `Lasso` regressor after normalizing the feature space, using the given value as relative strength of the regularization term. Computes `alpha = args.lasso / len(train_features[0])` to ensure that the regularization term is in the same order of magnitude independent of the size of the feature space.
- `--random_forest`: *Random Forest regression* using a random forest with default parameters as given by sklearn.

In addition to this, the script accepts the following optional parameters:
- `-s` or `--seed`: The random seed to use for initializing the random number generator (important for nondeterministic regressors). If none is given, a different initialization is used in every call to the script.
- `--shuffled`: If this flag is set, the regression is not only performed on the correct targets, but also on the shuffled ones.
- `-e` or `--evaluation_features`: If this optional argument is given, the specified feature vectors are used for testing. If not, then the `features.pickle` passed to the program will be used for both training and testing.

The script performs a cross-validation based on the fold structure given in `folds.csv`, where all augmented images that are based on the same original image belong into the same fold. The script reports MSE, MED (the mean Euclidean distance between the predicted points and the targets points), and the coefficient of determination R² in the output csv file for both the training and the test phase.

#### 3.2.5 Average Results over Folds

The script `regression.py` automatically performs an internal cross-validation and only reports the averaged results. However, if the neural network, on which the feature vectors are based, is also trained in a cross-validation scheme (as it is the case in the Shapes study), `regression.py` will be invoked once for each of the different network versions. In order to aggregate the results over these ``outer folds'', one can use the script `average_folds.py` as follows:
```
python -m code.ml.regression.average_folds input_path_template n_folds path/to/output.csv
```
The script uses the given `input_path_template` and the given number of folds `n_folds` to generate the paths to all individual csv files (e.g., if `ìnput_path_template` is `path/to/{0}/results.csv` and `n_folds` is `5`, then it will look for the files `path/to/0/results.csv`, `path/to/1/results.csv`, ..., `path/to/4/results.csv`). These individual csv files (which have been produced by `regression.py`) are read and the corresponding evaluation results are averaged across the different files. The aggregated results are then stored in `output.csv`.

### 3.3 Hybrid ANN

As a more complex approach, we investigated the usage of a hybrid ANN architecture which is trained on the tasks of classification, reconstruction, and/or mapping. All relevant scripts are contained in the `code/ml/ann` folder.

#### 3.3.1 Training and Evaluating the ANN

The script `run_ann.py` can be used to train and evaluate our proposed hybrid ANN architecture. It can be invoked as follows:
```
python -m code.ml.ann.run_ann path/to/Shapes.pickle path/to/Additional.pickle path/to/Berlin.pickle path/to/Sketchy.pickle path/to/targets.pickle space_name path/to/image_folder path/to/dissimilarities.pickle path/to/output.csv
```
Here, `Shapes.pickle`, `Additional.pickle`, `Berlin.pickle`, and `Sketchy.pickle` are the pickle files created by `prepare_Shapes_data.py`. Moreover, `targets.pickle` is the file generated by `prepare_targets.py` and `space_name` is the name of a target space contained in this file. The given `image_folder` contains all images of the Shapes study and `dissimilarities.pickle` is the output of `aggregate_similarities.py` and contains the target dissimilarity ratings. Both of these arguments are used to compute the correlation between the bottleneck layer activations and the dissimilarity ratings. Finally, `output.csv` is the file in which the results will be stored (the script appends to the file if it already exists).

In order to specify the training objective, the following three arguments can be used:
- `-c` or `--classification_weight`: Relative weight of the classification objective in the overall loss function.
- `-r` or `--reconstruction_weight`: Relative weight of the reconstruction objective in the overall loss function.
- `-m` or `--mapping_weight`: Relative weight of the mapping objective in the overall loss function.
Please note that all three weights default to zero, but that they need to be set in such a way that they sum to one.

The network can be regularized by using the following optional arguments:
- `-b` or `--bottleneck_size`: The number of units in the bottleneck layer, defaults to 512.
- `-w` or `--weight_decay_encoder`: The weight decay penalty used for weights in the encoder network. Defaults to 0.0005.
- `-v` or `--weight_decay_decoder`: The weight decay penalty used for weights in the decoder network. Defaults to 0.
- `-e` or `--encoder_dropout`: If this flag is set, dropout will be used in the first fully connected layer of the encoder.
- `-d` or `--decoder_dropout`: If this flag is set, dropout will be used in the first two fully connected layers of the decoder.
- `-n` or `--noise_prob`: The probability for the salt and pepper noise being applied to the inputs. Defaults to 0.1 (i.e., an expected amount of 10% of the pixels)
- `--bottleneck_dropout`: If this flag is set, dropout is also used in the bottleneck layer.
- `--noise_only_train`: If this flag is set, salt and pepper noise is only applied during training, but not during validation or test.

Moreover, one can pass the following optional arguments:
- `-s` or `--seed`: Seeds the random number generator with the given seed in order to make the results deterministic.
- `-t` or `--test`: If this flag is set, the number of iterations is drastically reduced for testing and debugging purposes.
- `-f` or `--fold`: Determines which fold to use for testing (defaults to 0).
- `--walltime`: Specifies the walltime in seconds before the job will be killed (relevant for grid execution). The script will try to stop its training before running over the walltime and store the current network weights in `data/Shapes/ml/experiment_N/snapshots/` as an hdf5 file.
- `--stopped_epoch`: Gives the epoch in which the last training was stopped. Load the model from `data/Shapes/ml/experiment_N/snapshots` and continue training with the next epoch (instead of starting from zero again).
- `--early_stopped`: If this flag is set, training was ended with early stopping. Load the model from `data/Shapes/ml/experiment_N/snapshots`, but do not continue training. Rather, switch to evaluation mode instead.
- `--optimizer`: Define the optimizer to use (`SGD` or `adam`, defaults to `adam`).
- `--learning_rate`: Initial learning rate for the optimizer, defaults to 0.0001.
- `--momentum`: Weight of the momentum term for the optimizer, defaults to 0.9.
- `--epochs`: Maximal number of epochs, defaults to 100.
- `--patience`: Patience for early stopping (i.e., number of epochs wating for improvement before terminating training), defaults to 10.
- `--padding`: Padding type for convolutions and max pooling when size reduction takes place. `valid` or `same`, defaults to `valid`.
- `--large_batch`: If this flag is set, training uses a batch size of 256 instead of 128.
- `--initial_stride`: Stride of the initial convolution, defaults to 2.
- `--image_size`: Size of the quadratic input image in pixels per dimension, defaults to 128.


Each execution of `run_ann.py` appends one line to the given `output.csv` file, representing the results for the given test fold. The first column of `output.csv` encodes the overall setup used with a single signature string and the second column gives the number of the test fold. The remaining columns contain the following information: 
- `kendall_DISTANCE_WEIGHTS`: In these columns, the kendall correlation between the bottleneck layer's activations and the dissimilarity ratings are reported. Here, `DISTANCE` gives the distance measure (Euclidean, Manhattan, or InnerProduct) and `WEIGHTS` indicates whether uniform or optimized weights were used.
- `loss`: Value of the overall loss function.
- `classification_loss`: Value of the classification objective (i.e., the categorical cross-entropy loss).
- `berlin_loss`: Value of the classification objective (i.e., the categorical cross-entropy loss) for the subset of TU Berlin data points and classes.
- `sketchy_loss`: Value of the classification objective (i.e., the categorical cross-entropy loss) for the subset of Sketchy data points and classes.
- `mapping_loss`: Value of the mapping objective (i.e., the MSE).
- `reconstruction_loss`: Value of the reconstruction objective (i.e., the binary cross-entropy loss).
- `berlin_weighted_acc`: The accuracy obtained with the classification output wrt TU Berlin data points and classes.
- `sketchy_weighted_acc`: The accuracy obtained with the classification output wrt Sketchy data points and classes.
- `mapping_weighted_med`: The mean Euclidean distance of the mapping task.
- `mapping_weighted_r2`: The coefficient of determination of the mapping task.

In order to run a five-fold cross-validation, one therefore needs to execute `run_ann.py` five times, using a different test fold number for each call, and aggregating the results afterwards. The script will furthermore at the end of the evaluation create an hdf5 file in `data/Shapes/ml/snapshots` with the final configuration of the model.

#### 3.3.2 Extracting Bottleneck Layer Activations

You can use the script `get_bottleneck_activations.py` to obtain the activations of the bottleneck layer for a given pre-trained instance of the ANN. It can be invoked as follows:
```
python -m code.ml.ann.get_bottleneck_activations path/to/Shapes.pickle path/to/model.h5 path/to/output.pickle
```
The script loads the model from `model.h5` (output of `run_ann.py`) and passes all images from `Shapes.pickle` (output of `prepare_Shapes_data.py`) through it. It stores the bottleneck activations is the file `output.pickle` in a format analogous to the one provided by `ann_features.py` to allow for further processing by `regression.py`.

The script accepts the following optional parameters:
- `-m` or `--mapping_used`: This flag must be set if the model was trained with a mapping weight greater than zero (otherwise the bottleneck activation cannot be correctly extracted).
- `-s` or `--seed`: Seeds the random number generator with the given seed in order to make the results deterministic.
- `-n` or `--noise_level`: Specifies the level of salt and pepper noise to apply to the images (defaults to 0.0).
- `-i` or `--image_size`: Size of the input image in pixels, defaults to 128.

#### 3.3.3 Average Results over Folds

The ANN script outputs one result line for each individual fold. In order to average the results across folds, you can invoke the script `average_folds.py` as follows:
```
python -m code.ml.ann.average_folds path/to/input.csv path/to/output.csv
```
The script goes through the given `input.csv` file (produced by `run_ann.py`) and averages all evaluation columns over the different folds, storing the results in the same manner in `output.csv` (only removing the `fold` column).

#### 3.3.4 Visualizing Reconstructions

In order to visualize the reconstructions of the autoencoder, you can use the script `visualize_reconstructions.py` as follows:
```
python -m code.ml.ann.visualize_reconstructions path/to/model.hdf5 path/to/input_image.png path/to/output_image.png
```
The script loads the given `ìnput_image`, resizes it, adds some noise (if desired), puts it through the autoencoder model stored in `model.hdf5`, and stores the result as `output_image`. It accepts the following optional parameters:
- `-s` or `--seed`: Seeds the random number generator with the given seed in order to make the results of the noise generation deterministic.
- `-n` or `--noise_level`: Specifies the level of salt and pepper noise to apply to the image before autoencoding (defaults to 0.0).
- `-i` or `--image_size`: Size of the network input image in pixels, defaults to 128.

