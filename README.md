# LearningPsychologicalSpaces
The code in this repository explores learning a mapping from images to psychological similarity spaces with neural networks.

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

One can run both the baselines and the regression multiple times in order to average over their respective performance. Both the baseline and the regression script will append their results if the output file already exists. In order to aggregate over these values, one can use the script `inception/collect_regression_results.py`. It takes the directory to run on as its single parameter. For each file in this directory, it averages over all rows and puts the resulting average RMSE into a file `summary.csv` in this directory.

