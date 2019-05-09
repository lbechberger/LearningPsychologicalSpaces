# LearningPsychologicalSpaces
v0.1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1220053.svg)](https://doi.org/10.5281/zenodo.1220053)

The code in this repository explores learning a mapping from images to psychological similarity spaces with neural networks. 

Research based on the code in this repository has been submitted to [AIC 2018](http://aic2018.pa.icar.cnr.it/): 

Lucas Bechberger and Elektra Kypridemou. "Mapping Images to Psychological Similarity Spaces Using Neural Networks" [Preprint](https://arxiv.org/abs/1804.07758)

## 0 About

Our scripts use TensorFlow 1.10 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project.

The folder `code` contains all python scripts used in our experiments. The usage of these scripts is detailed below.
The folder `data` contains the data used for the NOUN study inside the `NOUN` subfolder. This includes the dissimilarity ratings, the images, as well as all intermediate products created by our scripts. In the subfolder `Shapes` we will at some point add the respective results for the Shape study.

Our training data are the images and similarity ratings of the NOUN database (http://www.sussex.ac.uk/wordlab/noun), which were kindly provided by Jessica Horst and Michael Hout: 
Horst, Jessica S., and Michael C. Hout. "The Novel Object and Unusual Name (NOUN) Database: A collection of novel images for use in experimental research." Behavior research methods 48.4 (2016): 1393-1409.

The scripts `code/pipeline_NOUN.sh` and `code/pipeline_Shapes.sh` automatically execute all steps of our analysis pipeline and can be used both to reproduce our results and to see how the individual python scripts are actually executed in practice. The scripts `code/clean_NOUN.sh` and `code/clean_Shapes.sh` can be used to remove all temporary files and to start from a clean slate.

## 1 Preprocessing

The folder `code/preprocessing` contains various scripts for preprocessing the data set in order to prepare it for multidimensional scaling.

### 1.1 Parsing NOUN CSV file

The script `preprocess_NOUN.py` reads in the distances obtained via SpAM and stores them in a pickle file for further processing. It can be executed as follows from the project's root directory:
```
python code/preprocessing/preprocess_NOUN.py path/to/distance_table.csv path/to/output.pickle
```
The script reads the distance information from the specified csv file and stores it together with some further information in the specified pickle file. 

The resulting `output.pickle` file contains a dictionary with the follwing elements:
- `'categories'`: A dictionary using category names as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'visSim'`: Is the category visually homogeneous? 'Sim' means homogeneous, 'Dis' means not homogeneous, and 'x' means unclear.
  - `'artificial'`: Does the category consist of natural ('nat') or artificial ('art')?
  - `'items'`: A list of the IDs of all items that belong into this category. 
- `'items'`: A dictionary using item IDs as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'name'`: A human readable name for the item.
  - `'category'`: The name of the category this item belongs to.
- `'similarities'`: A dictionary using the string representation of sets of two items as keys and dictionaries as values. These dictionaries have the following elements:
  - `'relation'`: Is this a 'within' category or a 'between' category rating?
  - `'values'`: A list of similarity values (integers from 1 to 5, where 1 means 'no visual similarity' and 5 means 'high visual similarity')
  - `'border'`: An integer indicating the border between similarity ratings from the two studies. You can use `values[:border]` to access only the similarity ratings of the first study (only within category) and `values[border:]` to acces only the similarity ratings of the second study (mostly between cateory, but also some within category).

### 1.2 Parsing Shapes CSV Files

In order to make the Shapes data processible by our scripts, please run the script `preprocess_Shapes.py` as follows from the project's root directory:
```
python code/preprocessing/preprocess_Shape.py path/to/within.csv path/to/within_between.csv path/to/output.pickle
```

The file `within.csv` contains within category similarity judments, the file `within_between.csv` contains similarity ratings both within and between categories. All of these similarity ratings are based on shape similarity only (participants were explicitly asked to not consider *conceptual* similarity).

The resulting `output.pickle` file contains a dictionary with the follwing elements:
- `'categories'`: A dictionary using category names as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'visSim'`: Is the category visually homogeneous? 'Sim' means homogeneous, 'Dis' means not homogeneous, and 'x' means unclear.
  - `'artificial'`: Does the category consist of natural ('nat') or artificial ('art')?
  - `'items'`: A list of the IDs of all items that belong into this category. 
- `'items'`: A dictionary using item IDs as keys and containing dictionaries as values. These dictionaries have the following elements:
  - `'name'`: A human readable name for the item.
  - `'category'`: The name of the category this item belongs to.
- `'similarities'`: A dictionary using the string representation of sets of two items as keys and dictionaries as values. These dictionaries have the following elements:
  - `'relation'`: Is this a 'within' category or a 'between' category rating?
  - `'values'`: A list of similarity values (integers from 1 to 5, where 1 means 'no visual similarity' and 5 means 'high visual similarity')
  - `'border'`: An integer indicating the border between similarity ratings from the two studies. You can use `values[:border]` to access only the similarity ratings of the first study (only within category) and `values[border:]` to acces only the similarity ratings of the second study (mostly between cateory, but also some within category).

### 1.3 Aggregating Similarity Ratings

The next step in the preprocessing pipeline is to extract similarity ratings from the overall data set. This can be done with the script `compute_similarities.py`. You can execute it as follows from the project's root directory:
```
python code/preprocessing/compute_similarities.py path/to/input_file.pickle path/to/output_file.pickle
```

The first argument to this script should be the output file generated by `preprocess.py`, the second parameter determines where the resulting similarity values are stored. After converting the data from the pickle file provided by `preprocess.py` into a dissimilarity matrix, some information about this matrix is printed out.

The script takes the following optional arguments:
- `-s` or `--subset`: Specifies which subset of the similarity ratings to use. Default is `all` (which means that all similarity ratings from both studies are used). Another supported option is `between` where only the ratings from the second study (found in `within_between.csv`) are used. Here, all items that did not appear in the second study are removed from the dissimilarity matrix. A third option is `cats` which only considers the categories used in the second study, but which keeps all items from these categories (also items that were only used in the first, but not in the second study). The fourth option `within` only uses data from the first study.
- `-m` or `--median`: Use the median instead of the mean for aggregating the similarity ratings across participants
- `-l` or `--limit`: Limit the number of similarity ratings to use to ensure that an equal amount of ratings is aggregated for all item pairs. Use the minimal number of ratings observed for any item pair as limit.
- `-p` or `--plot`: Plot some histogram of the similarity values and store it in the same folder where also the output pickle file is stored.

The result is a pickle file which consists of a dictionary with the following content:
- `'items'`: The list of item-IDs of all the items for which the similarity values have been computed
- `'item_names'`: The list of item names for all the items (sorted in same way as `'items'`).
- `'similarities'`: A quadratic matrix of similarity values. Both rows and columns are ordered like in `'items'`. Values of `nan` are used to indicate that there is no similarity rating available for a pair of stimuli.
- `'dissimilarities'`: A quadratic matrix of dissimilarity values analogous to `'similarities'`. Here, values of 0 indicate missing similarity ratings.

### 1.4 Analyzing Similarity Ratings

The script `analyze_similarities.py` can be used to collect some statistics on the distribution of similarity ratings for a given subset of the data (prints out some statistics and creates some plots). It can be executed as follows:
```
python code/preprocessing/analyze_similarities.py path/to/input_file.pickle
```
The input file is here the `output.pickle` created by the `preprocess_Shapes.py` script. The script takes two optional parameters:

- `-s` or `--subset`: Specifies which subset of the similarity ratings to use. Default is `all` (which means that all similarity ratings from both studies are used). Another supported option is `between` where only the ratings from the second study (found in `within_between.csv`) are used. Here, all items that did not appear in the second study are removed from the dissimilarity matrix. A third option is `cats` which only considers the categories used in the second study, but which keeps all items from these categories (also items that were only used in the first, but not in the second study).
- `-o` or `--output_path`: The path to the folder where the plots shall be stored. Defaults to `.`, i.e., the current working directory.

### 1.5 Creating Average Images

The script `average_images.py` can be used in order to create an average image for each of the categories. It can be invoked as follows:
```
python code/preprocessing/average_images.py path/to/input_file.pickle path/to/image_folder
```
Here, `image_file.pickle` corresponds to the output file of `preprocess_Shapes.py`. The script takes the following optional arguments:
- `-o` or `--output_folder`: The destination folder for the output images, defaults to `.`, i.e., the current working directory.
- `-r` or `--resolution`: The desired size (width and height) of the output images, defaults to 283 (size of the original images).
- `-s` or `--subset`: The subset of data to use, defaults to `all`. Possible other options are `between`, `within`, and `cats`.

### 1.6 Writing CSV Files of Aggregated Dissimilarities
The R script for MDS needs the aggregated dissimilarity data in form of a CSV file. The script `pickle_to_csv.py` stores the similaritiy ratings from `input_file.pickle` into a CSV file called `distance_matrix.csv` as well as the list of item names in a file called `item_names.csv`. Both output files are stored in the given `output_folder`. The `input_file.pickle` should be the file created by `compute_similarities.py`. The script can be invoked as follows:
```
python code/preprocessing/pickle_to_csv.py path/to/input_file.pickle path/to/output_folder/
```

## 2 Multidimensional Scaling

The folder `code/mds` contains various scripts for transforming the given data set from pairwise similarity ratings into a conceptual space and for analyzing the resulting space.

### 2.1 Applying MDS

The script `mds.r` runs four different versions of multidimensional scaling based on the implementations in R. More specifically, it uses the Eigenvalue-based classical MDS [cmdscale](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html), Kruskal's nonmetric MDS [isoMDS](https://stat.ethz.ch/R-manual/R-devel/library/MASS/html/isoMDS.html), and both metric and nonmetric SMACOF [smacofSym](https://cran.r-project.org/web/packages/smacof/smacof.pdf#page.55). For Kruskal's algorithm and for SMACOF, multiple random starts are used and the best result is kept. You can execute the script as follows from the project's root directory:
```
Rscript code/mds/mds.r -d path/to/distance_matrix.csv -i path/to/item_names.csv -o path/to/output/directory
```
Here, `distance_matrix.csv` is a CSV file which contains the matrix of pairwise dissimilarities and `item_names.csv` contains the item names (one name per row, same order as in the distance matrix). The resulting vectors are stored in the given output directory. All three of these arguments are mandatory. Moreover, a CSV file is created in the output directory, which stores both metric and nonmetric stress for each of the spaces. 

The script takes the following optional arguments:
- `-k` or `--dims`: Specifies the maximal number of dimensions to investigate. Default value is 20, which means that the script will run the MDS algorithm 20 times, obtaining spaces of dimensionality 1 to 20.
- `-n` or `--n_init`: Specifies how often Kruskal's algorithm is restarted with a new random initialization. Of all of these runs, only the best result (i.e., the one with the lowest resulting stress) is kept. Default value here is 64.
- `m` or `--max_iter`: Specifies the maximum number of iterations computed within the SMACOF algorithm. Default values is 1000.
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the results deterministic. If no seed is given, then the random number generator is not seeded.
- `--metric`: If this flag is set, *metric* MDS is used instead of *nonmetric* MDS (which is the default).
- `--smacof`: If this flag is set, the SMACOF algorithm is used. If not set, then classical MDS or Kruskal's algorithm are used.
- `-t` or `--tiebreaker`: Specifies the type of tie breaking used in the SMACOF algorithm (possible values: `primary`, `secondary`, `tertiary`, default: `primary`).

We implemented the MDS step in R and not in Python because R offers a greater variety of MDS algorithms. Moreover, nonmetric SMACOF with Python's `sklearn` library produced poor results which might be due to a programming bug.

### 2.2 Normalizing the Resulting Spaces

In order to make the individual MDS solutions more comparable, we normalize them by moving their centroid to the origin and by making sure that their root mean squared distance to the origin equals one. This is done by the script `normalize_spaces.py`, which can be invoked by simply giving it the path to the directory containing all the vector files:
```
python code/mds/normalize_spaces.py path/to/input_folder
```
The script **overrides** the original files. It can take the following optional arguments:
- `-b` or `--backup`: Create a backup of the original files. Will be stored in the same folder as the original files, file name is identical, but 'backup' is appended before the file extension.
- `-v` or `--verbose`: Prints some debug information during processing (old centroid, old root mean squared distance to origin, new centroid, new root mean squared distance to origin).

**It is important to run this script before using the MDS spaces for the regression task -- only by normalizing the spaces, we can make sure that the (R)MSE values are comparable across spaces!**

### 2.3 Visualizing the Resuling Spaces

The script `visualize.py` can be used to create two-dimensional plots of the resulting MDS spaces. You can execute it as follows from the project's root directory:
```
python code/mds/visualize.py path/to/vector_folder path/to/output_folder/
```
The script reads in the vectors from all csv files in the `vector_folder`, creates two-dimensional plots for all pairs of dimensions, and stores them in the given output folder. 

The script takes the following optional arguments:
- `-i` or `--image_folder`: Path to a folder where the images of the items are stored. If this is given, then the images from this folder are used in the visualization. If no image folder is given, then data points are labeled with their item ID.
- `-z` or `--zoom`: Determines how much the images are scaled. Default is 0.15.

### 2.4 Checking for Convexity

The script `analyze_convexity.py` can be used to check whether the categories within the space are convex. This is only applicable to the Shapes data set, as there are no categories in NOUN. The script iterates over all categories, builds a convex hull of the items belonging to this category and counts how many points from other categories lie within this convex hull. Each point that lies in the convex hull of a different concept is counted as one violation. The script outputs the number of violations for each category, together with an estimate of how many violations would be expected if points are randomly sampled from a uniform distribution, a normal distribution, or the overall set of given points.

The script finally outputs the total number of violations as well as the group-wise number of violations for visual similarity and for natrualness. For the latter two, all four possible combination of classes are analyzed. An output for the pair `Sim`-`Dis` gives the number of violations observed where items from a category which is visually dissimilar was found to lie within the convex hull of another category which is visually similar, i.e., the first element gives the type of category used for building the convex hull, whereas the second element gives the type of category of the "intruder" items.

The script can be exectued as follows (where `n_dims` is the number of dimension of this specific MDS space):
```
python code/mds/analyze_convexity.py path/to/vectors.csv path/to/data.pickle n_dims
```
It takes the following optional arguments:
- `-o` or `--output_file`: If an output file is given, the results are appended to this file in CSV style.
- `-b` or `--baseline`: Ony if this flag is set, the script will also estimate the expected values of randomly drawn points.
- `-r` or `--repetitions`: Determines the number of repetitions used when sampling from the baselines. Defaults to 20. More samples means more accurate estimation, but longer runtime.


### 2.5 Searching for Interpretable Directions
The script `check_interpretability.py` tries to find interpretable directions in a given MDS space based on prior binary classifications of the items. The script iterates over all files in the classification folder and constructs a classification problem for each of these files. Each file is expected to contain a list of positive examples, represented by one item ID per line. A linear SVM is trained using the vectors provided in the csv file and the classification as extracted from the classification file. All data points are used for both training and evaluating. Evaulation is done by using Cohen's kappa. The script outputs for each classification task the value of Cohen's kappa as well as the normal vector of the separating hyperplane. The latter can be thought of as an interpretable direction if the value of kappa is sufficiently high. Just like `analyze_convexity.py`, also the `check_interpretability.py` script compares the result to the average over multiple repetitions for randomly sampled points (uniformly distributed vectors, normally distributed vectors, shuffled assignment of real vectors).

The script can be invoked as follows (where `n_dims` is the number of dimensions of the underlying space):
```
python code/mds/check_interpretability.py path/to/vectors.csv path/to/classification/folder/ n_dims
```
It takes the following optional arguments:
- `-o` or `--output_file`: If an output file is given, the results are appended to this file in CSV style.
- `-b` or `--baseline`: Ony if this flag is set, the script will also estimate the expected values of randomly drawn points.
- `-r` or `--repetitions`: Determines the number of repetitions used when sampling from the baselines. Defaults to 20. More samples means more accurate estimation, but longer runtime.


## 3 Correlations to Similarity Ratings

The folder `correlations` contains scripts for estimating how well the MDS spaces represent the underlying similarity ratings. As a baseline, pixel-based similarities of the corresponding images are used.

### 3.1 Pixel-Based Similarities

The script `image_correlations.py` loads the images and interprets them as one-dimensional vectors of pixel values. It then computes for each pair of items the different similarity measures (i.e., cosine distance, Euclidean distance, Manhattan distance) of their pixel-based representation. The resulting similarity matrix is compared to the one obtained from human similarity judgements by computing different correlation statistics (Pearson's R, Spearman's Rho, Kendall's Tau, and the coefficient of determination R Squared). The script can be executed as follows, where `similarity_file.pickle` is the output file of the overall preprocessing and where `image_folder` is the directory containing all images:
```
python code/correlations/image_correlations.py path/to/similarity_file.pickle path/to/image_folder
```
In addition to doing these computations on the full pixel-wise information, the script also shrinks the image by aggregating all pixels within a block of size `k` times `k` into a single number. The script iterates over all possible sizes of k (from 1 to 283 in our case) and uses different aggregation strategies (namely: max, min, std, var, median, product).
The script takes the following optional parameter:
- `-o` or `--output`: The output folder where the resulting correlation values are stored (default: `.`, i.e., the current working directory).
- `-s` or `--size`: The size of the image, i.e., the maximal number of `k` to use (default: 283).
- `-g` or `--greyscale`: If this flag is set, the three color channels are collapsed into a single greyscale channel when loading the images. If not, full RGB information is used.
- `-p` or `--plot`: If this flag is set, scatter plots are created and stored for each of the similarity measures.

### 3.2 MDS-Based Similarities

The script `mds_correlations.py` loads the MDS vectors and derives distances between pairs of stimuli based on the cosine distance, the Euclidean distance, and the Manhattan distance. These distances are then correlated to the human dissimilarity ratings with Pearson's R, Spearman's Rho, Kendall's Tau, and the coefficient of determination R Squared. The script can be executed as follows:
```
python code/correlations/mds_correlations.py path/to/similarity_file.pickle path/to/mds_folder
```
Here, `similarity_file.pickle` is again the output file of the overall preprocessing, whereas `mds_folder` is the folder where the MDS vectors are stored. The script takes the following optional arguments:
- `-o` or `--output`: The output folder where the resulting correlation values are stored (default: `.`, i.e., the current working directory).
- `--n_min`: The size of the smallest space to investigate (defaults to 1).
- `--n_max`: The size of the largest space to investigate (defaults to 20).
- `-p` or `--plot`: If this flag is set, scatter plots are created and stored for each of the similarity measures.

### 3.3 Visualizing The Correlations

The script `visualize_correlations.py` can be used to visualize the results of the correlation computations. It can be invoked as follows:
```
python code/correlations/visualize_correlations.py path/to/pixel_file.csv path/to/mds_file.csv
```
Here, `pixel_file.csv` and `mds_file.csv` are the output files of `image_correlations.py` and `mds_correlations.py`, respectively. The script takes the following additional optional arugments:
- `-o` or `--output`: The output folder where the resulting visualizations are stored (default: `.`, i.e., the current working directory).


## 4 Regression with Inception-v3

In order to run a linear regression based on the features extracted by the inception-v3 network, multiple processing steps are necessary. Firstly, we need to augment our data set by creating a large amount of slightly distorted image variants. This is done in order to achieve a data set of reasonable size for a machine learning task. Afterwards, each of these images need to be mapped to a feature vector and to a corresponding ground truth MDS vector. In a final step, the linear regression as well as different baselines are executed. All scripts are contained in the `code/inception` folder.

### 4.1 Data Augmentation

We used [ImgAug](https://github.com/aleju/imgaug) for augmenting our image data set. This is done with the script `data_augmentation.py`. It can be invoked as follows:
```
python code/inception/data_augmentation.py path/to/image_folder/ path/to/output_folder/ n
```
The script searches for all jpg images in the given `image_folder`, creates `n` augmented samples of each image and stores the results in the given `output_folder` (one pickle file per original image). The script takes the following optional command line arguments:
- `-s` or `--seed`: Specify a seed for the random number generator in order to make the results deterministic. If no seed is given, then the random number generator is not seeded.
- `-i` or `--image_size`: The expected image size in pixels. Defaults to 300.

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

### 4.2 Visualizing Augmented Images

In order to visually check that the augmentation step worked, you can use the script `show_augmented_images.py` to display them. It can be executed as follows:
```
python code/inception/show_augmented_images.py path/to/augmented.pickle
```
Here, `augmented.pickle` is one of the pickle files created by `data_augmentation.py`. By default, the script displays three rows (adjustable via `-r` or `--rows`) and four columns (adjustable via `-c` or `--columns`).

### 4.3 Feature Extraction with Inception-v3

In order to create feature vectors based on the inception-v3 network, one can use the script `inception/create_feature_vectors.py`. It is invoked as follows:
```
python code/inception/create_feature_vectors.py path/to/model_folder path/to/input_folder path/to/output.pickle
```
The script downloads the [Inception-v3 network](https://arxiv.org/abs/1512.00567) into the folder specified by `model_folder`, reads all augmented images from the folder specified by `input_folder`, uses them as input to the inception network, grabs the activations of the second-to-last layer of the network (2048 neurons) and stores a dictionary mapping from image name to a list of feature vectors in the pickle file specified by `output.pickle` (defaults to `inception/features/features`).

### 4.4 Feature Extraction by Downscaling Images
**TODO**

### 4.5 Cluster Analysis
**TODO**

### 4.6 Defining Regression Targets

**Currently outdated, will be updated soon**
In order to see whether the organization of points within the MDS space is meaningful and learnable by machine learning, we also shuffle the assignments of images to points with the script `inception/shuffle_targets.py`. It takes as parameters the filename of the original mapping and the filename for the shuffled mapping: `python shuffle_targets.py in_file_name out_file_name`. The SGE script `run_shuffler.sge` can be used to submit this script to a Sun grid engine.

### 4.7 Baselines

**Currently outdated, will be updated soon**
We have programmed four simple baselines that can be run by executing the script `inception/baselines.py`. The script expects a single parameter, which is the name of the configuration inside the `grid_search.cfg` to use. This configuration should contain information on the feature vectors file, on the file containing the mapping from images to points in the MDS space, and on the dimensionality of the MDS space. The script computes the RMSE for each of the baselines in an image-ID based leave-one-out procedure (i.e., all augmented images based on one original image are used as test set each time) and stores the results in separate files in the folder `inception/regression` (which has to be existent before the script is run). The baselines are as follows:
- *Zero baseline*: Always predict the origin (i.e., a vector where all entries are zero)
- *Mean baseline*: Always predict the mean of the target vectors seen during training.
- *Distribution baseline*: Estimate a Gaussian distribution based on the target vectors seen during training. For making a prediction, draw a random sample from this distribution.
- *Random draw baseline*: Use a randomly selected target vectors from the training set as prediction.

The SGE script `run_baselines.sge` can be used to submit this script to a Sun grid engine. It takes two parameters: The configuration name to pass to the python script and the number of repetitions.

### 4.8 Regression
**Currently outdated, will be updated soon**
The script `inception/run_sklearn_regression.sge` uses the linear regression of scikit-learn on the features extracted by the inception network. Like the baseline script, it takes a single parameter, which is the name of the configuration inside the `grid_search.cfg` to use. Again, an image-ID based leave-one-out procedure is used to compute the RMSE. The result is stored in a file in the directory `inception/regression` (which has to be existent before the script is run). 


