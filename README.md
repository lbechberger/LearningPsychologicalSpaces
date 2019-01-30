# LearningPsychologicalSpaces
v0.1: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1220053.svg)](https://doi.org/10.5281/zenodo.1220053)

The code in this repository explores learning a mapping from images to psychological similarity spaces with neural networks. 

The research based on the code in this repository has been submitted to [AIC 2018](http://aic2018.pa.icar.cnr.it/): 
Lucas Bechberger and Elektra Kypridemou. "Mapping Images to Psychological Similarity Spaces Using Neural Networks" [Preprint](https://arxiv.org/abs/1804.07758)

## About

Our scripts use TensorFlow 1.4.1 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project. **Please note that all python scripts have to be executed from the directory in which they reside in order to work properly.**

The folder `NOUN` contains a first feasibility study on the NOUN database (presented at AIC 2018), conducted by Lucas Bechberger and Elektra Kypridemou.
The folder `Shapes` contains a follow-up study on a data set of shapes (work in progress), conducted by Lucas Bechberger and Margit Scheibel.

Additional `README.md` files in those folders give more detail about the respective project and its scripts.

# NOUN

## About

Our scripts use TensorFlow 1.4.1 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project. **Please note that all python scripts have to be executed from the directory in which they reside in order to work properly.**

## Data
Our training data are the images and similarity ratings of the NOUN database (http://www.sussex.ac.uk/wordlab/noun): 
Horst, Jessica S., and Michael C. Hout. "The Novel Object and Unusual Name (NOUN) Database: A collection of novel images for use in experimental research." Behavior research methods 48.4 (2016): 1393-1409.

## Multi-dimensional Scaling

The code necessary to run the multi-dimensional scaling can be found in the `mds` directory. It uses the similarity ratings available with the NOUN database and applies the MDS algorithm from scikit-learn in order to derive a spatial representation. You can invoke it by simply calling `python MDS-NOUN-4d.py` from within the `mds` directory. The script generates an illustration of the resulting space as well as a csv file where the first column identifies the image by its ID and the remaining columns provide the coordinates of this image in the MDS space.

## Data Augmentation

We used [ImgAug](https://github.com/aleju/imgaug) for augmenting our image data set. This is done with the script `inception/data_augmentation.py`. By default it will search for jpg images in a folder called `images` in the project's main directory, create 1000 samples per original image and store the results in the folder `inception/features/aumgented/`. This behavior can be adjusted by the flags `--images_dir`, `--output_dir`, and `--n_samples`, respectively. After running this script, you will have one pickle file per original image in the specified output folder, containing all the augmented samples for this image. The SGE script `run_augmentation.sge` can be used to submit this script to a Sun grid engine.

## Creating the feature vectors from the Inception-v3 network

The script `inception/create_feature_vectors.py` downloads the [Inception-v3 network](https://arxiv.org/abs/1512.00567) into the folder specified by `--model_dir` (defaults to `/tmp/imagenet/`), reads all augmented images from the folder specified by `--input_dir` (defaults to `inception/features/augmented`), uses them as input to the inception network, grabs the activations of the second-to-last layer of the network (2048 neurons) and stores them as feature vectors in the folder specified by `--output_dir` (defaults to `inception/features/features`). Again, there will be one file per original image, containing the feature vectors for all augmented images that were based on the same original image. The SGE script `run_feature_extraction.sge` can be used to submit this script to a Sun grid engine.

Afterwards, all individual files are collected and aggregated in a single feature file (for convenience reasons) by the `inception/collect_feature_vectors.py` which takes two arguments: Its first argument is the path to the folger containing all individual feature files, its second argument is the path to the folder where to store the aggregated file in pickle format (which will be named `images`). The SGE script `run_feature_collection.sge` can be used to submit this script to a Sun grid engine.

## Shuffling the target vectors

In order to see whether the organization of points within the MDS space is meaningful and learnable by machine learning, we also shuffle the assignments of images to points with the script `inception/shuffle_targets.py`. It takes as parameters the filename of the original mapping and the filename for the shuffled mapping: `python shuffle_targets.py in_file_name out_file_name`. The SGE script `run_shuffler.sge` can be used to submit this script to a Sun grid engine.

## Baselines
We have programmed four simple baselines that can be run by executing the script `inception/baselines.py`. The script expects a single parameter, which is the name of the configuration inside the `grid_search.cfg` to use. This configuration should contain information on the feature vectors file, on the file containing the mapping from images to points in the MDS space, and on the dimensionality of the MDS space. The script computes the RMSE for each of the baselines in an image-ID based leave-one-out procedure (i.e., all augmented images based on one original image are used as test set each time) and stores the results in separate files in the folder `inception/regression` (which has to be existent before the script is run). The baselines are as follows:
- *Zero baseline*: Always predict the origin (i.e., a vector where all entries are zero)
- *Mean baseline*: Always predict the mean of the target vectors seen during training.
- *Distribution baseline*: Estimate a Gaussian distribution based on the target vectors seen during training. For making a prediction, draw a random sample from this distribution.
- *Random draw baseline*: Use a randomly selected target vectors from the training set as prediction.

The SGE script `run_baselines.sge` can be used to submit this script to a Sun grid engine. It takes two parameters: The configuration name to pass to the python script and the number of repetitions.

## Regression
The script `inception/run_sklearn_regression.sge` uses the linear regression of scikit-learn on the features extracted by the inception network. Like the baseline script, it takes a single parameter, which is the name of the configuration inside the `grid_search.cfg` to use. Again, an image-ID based leave-one-out procedure is used to compute the RMSE. The result is stored in a file in the directory `inception/regression` (which has to be existent before the script is run). 

The SGE script `run_sklearn_regression.sge` can be used to submit this script to a Sun grid engine. It takes two parameters: The configuration name to pass to the python script and the number of repetitions.

## Collecting the results

One can run both the baselines and the regression multiple times in order to average over their respective performance. Both the baseline and the regression script will append their results if the output file already exists. In order to aggregate over these values, one can use the script `inception/collect_regression_results.py`. It takes the directory to run on as its single parameter. For each file in this directory, it averages over all rows and puts the resulting average RMSE into a file `summary.csv` in this directory. The SGE script `run_regression_collection.sge` can be used to submit this script to a Sun grid engine.

# Shapes

## About

Our scripts use TensorFlow 1.4.1 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project. 

Please find more detailed instructions on how to use the scripts below in the respective sections of the workflow.

## Multidimensional Scaling

The folder `mds` contains various scripts for transforming the given data set from pairwise similarity ratings into a conceptual space.

### Preprocessing

In order to make the original data processible by our scripts, please run the script `preprocess.py` as follows from the project's root directory:
```
python mds/preprocess.py path/to/within.csv path/to/within_between.csv path/to/output.pickle
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

### Extracting Similarity Ratings

The next step in our pipeline is to extract similarity ratings from the overall data set. This can be done with the script `compute_similarities.py`. You can execute it as follows from the project's root directory:
```
python mds/compute_similarities.py path/to/input_file.pickle path/to/output_file.pickle
```

The first argument to this script should be the output file generated by `preprocess.py`, the second parameter determines where the resulting similarity values are stored. After converting the data from the pickle file provided by `preprocess.py` into a dissimilarity matrix, some information about this matrix is printed out.

The script takes the following optional arguments:
- `-s` or `--subset`: Specifies which subset of the similarity ratings to use. Default is `all` (which means that all similarity ratings from both studies are used). Another supported option is `between` where only the ratings from the second study (found in `within_between.csv`) are used. Here, all items that did not appear in the second study are removed from the dissimilarity matrix. A third option is `cats` which only considers the categories used in the second study, but which keeps all items from these categories (also items that were only used in the first, but not in the second study).
- `-m` or `--median`: Use the median instead of the mean for aggregating the similarity ratings across participants
- `-l` or `--limit`: Limit the number of similarity ratings to use to ensure that an equal amount of ratings is aggregated for all item pairs. Use the minimal number of ratings observed for any item pair as limit.
- `-p` or `--plot`: Plot some histogram of the similarity values and store it in the same folder where also the output pickle file is stored.

The resulting `output_file.pickle` contains a dictionary with the following elements:
- `'items'`: The list of item-IDs of all the items for which the similarity values have been computed
- `'item_names'`: The list of item names for all the items (sorted in same way as `'items'`).
- `'similarities'`: A quadratic matrix of similarity values. Both rows and columns are ordered like in `'items'`. Values of `nan` are used to indicate that there is no similarity rating available for a pair of stimuli.
- `'dissimilarities'`: A quadratic matrix of dissimilarity values (computed as `5 - similarity`) analogous to `'similarities'`. Here, values of 0 indicate missing similarity ratings.

### Analyzing Similarity Ratings

The script `analyze_similarities.py` can be used to collect some statistics on the distribution of similarity ratings for a given subset of the data (prints out some statistics and creates some plots). It can be exectuted as follows:
```
python mds/analyze_similarities.py path/to/input_file.pickle
```
The input file is here the `output.pickle` created by the `preprocess.py` script. The script takes two optional parameters:

- `-s` or `--subset`: Specifies which subset of the similarity ratings to use. Default is `all` (which means that all similarity ratings from both studies are used). Another supported option is `between` where only the ratings from the second study (found in `within_between.csv`) are used. Here, all items that did not appear in the second study are removed from the dissimilarity matrix. A third option is `cats` which only considers the categories used in the second study, but which keeps all items from these categories (also items that were only used in the first, but not in the second study).
- `-o` or `--output_path`: The path to the folder where the plots shall be stored. Defaults to `.`, i.e., the current working directory.


### MDS

The script `mds.py` runs the MDS algorithm provided by the `sklearn` library (see documentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)) which is based on the SMACOF algorithm. You can execute the script as follows from the project's root directory:
```
python mds/mds.py path/to/data.pickle
```
Here, `data.pickle` is the file created by `compute_similarities.py`.

The script takes the following optional arguments:
- `-n` or `--n_init`: Specifies how often the SMACOF algorithm is restarted with a new random initialization. Of all of these runs, only the best result (i.e., the one with the lowest resulting stress) is kept. Default value here is 4.
- `-d` or `--dims`: Specifies the maximal number of dimensions to investigate. Default value is 20, which means that the script will run the MDS algorithm 20 times, obtaining spaces of dimensionality 1 to 20.
- `-i` or `--max_iter`: Specifies the maximum number of iterations computed within the SMACOF algorithm. Default values is 300.
- `-e` or `--export`: If this flag is set and an export directory is given, then the created MDS vectors will be exported in csv files into the given export directory.
- `-p` or `--plot`: If this flag is set, a plot of stress over the number of dimensions is displayed to the user.

### Visualizing the Resuling Space

The script `visualize.py` can be used to create two-dimensional plots of the resulting space. You can execute it as follows from the project's root directory:
```
python mds/visualize.py path/to/vectors.csv path/to/output/folder/ n_dims
```
The script reads in the vectors from the `vectors.csv` file, creates two-dimensional plots for all pairs of dimensions, and stores them in the given output folder. `n_dims` tells the script how many dimensions there are in the space.

The script takes the following optional arguments:
- `-i` or `--image_folder`: Path to a folder where the images of the items are stored. If this is given, then the images from this folder are used in the visualization. If no image folder is given, then data points are labeled with their item ID.
- `-z` or `--zoom`: Determines how much the images are scaled. Default is 0.15.

### Checking for Convexity

The script `analyze_space.py` can be used to check whether the categories within the space are convex. The script iterates over all categories, builds a convex hull of the items belonging to this category and counts how many points from other categories lie within this convex hull. Each point that lies in the convex hull of a different concept is counted as one violation. The script outputs the number of violations for each category, together with an estimate of how many violations would be expected if points are randomly sampled from a uniform distribution, a normal distribution, or the overall set of given points.

The script can be exectued as follows (where `n_dims` is the number of dimension of this specific MDS space):
```
python mds/analyze_space.py path/to/vectors.csv path/to/data.pickle n_dims
```

### Searching for Interpretable Directions
The script `check_interpretablilites.py` tries to find interpretable directions in a given MDS space based on prior binary classifications of the items. It can be invoked as follows (where `n` is the number of dimensions of the underlying space):
```
python mds/check_interpretabilities.py path/to/vectors.csv path/to/classification/folder/ n
```
The script iterates over all files in the classification folder and constructs a classification problem for each of these files. Each file is expected to contain a list of positive examples, represented by one item ID per line. A linear SVM is trained using the vectors provided in the csv file and the classification as extracted from the classification file. All data points are used for both training and evaluating. Evaulation is done by using Cohen's kappa. The script outputs for each classification task the value of Cohen's kappa as well as the normal vector of the separating hyperplane. The latter can be thought of as an interpretable direction if the value of kappa is sufficiently high. Just like `analyze_space.py`, also the `check_interpretabilities.py` script compares the result to the average over 100 iterations for randomly sampled points (uniformly distributed vectors, normally distributed vectors, shuffled assignment of real vectors).

## Baselines

The folder `baseline` contains scripts for evaluating the performance of simple pixel-based baselines.

### Pixel-Based Similarities

The script `similarities.py` loads the images and interprets them as one-dimensional vectors of pixel values. It then computes for each pair of items the different similarity measures (i.e., cosine similarity, Euclidean distance, Manhattan distance, Mutual information) of their pixel-based representation. The resulting similarity matrix is compared to the one obtained from human similarity judgements by computing Pearson's r. The script can be executed as follows, where `similarity_file.pickle` is the output file produced by `compute_similarities.py` and where `image_folder` is the directory containing all images:
```
python baseline/cosine_similarity.py path/to/similarity_file.pickle path/to/image_folder
```
In addition to doing these computations on the full pixel-wise information, the script also shrinks the image by aggregating all pixels within a block of size k times k into a single number. The script iterates over all possible sizes of k (from 1 to 283 in our case) and uses different aggregation strategies (namely: max, min, std, var, median, product).
The script takes the following optional parameter:
- `-o` or `--output`: The output folder where the resulting correlation values are stored (default: `analysis`).
- `-s` or `--size`: The size of the image, i.e., the maximal number of k to use (default: 283).

