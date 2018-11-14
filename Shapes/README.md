# Shapes

## About

Our scripts use TensorFlow 1.4.1 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project. 

Please find more detailed instructions on how to use the scripts below in the respective sections of the workflow.

## Multidimensional Scaling

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

## MDS

The script `mds.py` runs the MDS algorithm provided by the `sklearn` library (see documentation [here](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)) which is based on the SMACOF algorithm. After converting the data from the pickle file provided by `preprocess.py` into a dissimilarity matrix, some information about this matrix is printed out. Aftwerwards, the MDS is computed for different numbers of dimensions. A plot displaying stress against the number of dimensions is generated and displayed to the user.

You can execute the script as follows from the project's root directory:
```
python mds/mds.py path/to/data.pickle
```

The script takes the following optional arguments:
- `-s` or `--subset`: Specifies which subset of the similarity ratings to use. Default is `all` (which means that all similarity ratings from both studies are used). Another supported option is `between` where only the ratings from the second study (found in `within_between.csv`) are used. Here, all items that did not appear in the second study are removed from the dissimilarity matrix. A third option is `cats` which only considers the categories used in the second study, but which keeps all items from these categories (also items that were only used in the first, but not in the second study).
- `-n` or `--n_init`: Specifies how often the SMACOF algorithm is restarted with a new random initialization. Of all of these runs, only the best result (i.e., the one with the lowest resulting stress) is kept. Default value here is 4.
- `-d` or `--dims`: Specifies the maximal number of dimensions to investigate. Default value is 20, which means that the script will run the MDS algorithm 20 times, obtaining spaces of dimensionality 1 to 20.
- `-i` or `--max_iter`: Specifies the maximum number of iterations computed within the SMACOF algorithm. Default values is 300.
- `-e` or `--export`: If this flag is set and an export directory is given, then the created MDS vectors will be exported in csv files into the given export directory.

## Visualizing the Resuling Space

The script `visualize.py` can be used to create two-dimensional plots of the resulting space. You can execute it as follows from the project's root directory:
```
python mds/visualize.py path/to/vectors.csv path/to/output/folder n_dims
```
The script reads in the vectors from the `vectors.csv` file, creates two-dimensional plots for all pairs of dimensions, and stores them in the given output folder. `n_dims` tells the script how many dimensions there are in the space.

The script takes the following optional arguments:
- `-i` or `--image_folder`: Path to a folder where the images of the items are stored. If this is given, then the images from this folder are used in the visualization. If no image folder is given, then data points are labeled with their item ID.
- `-z` or `--zoom`: Determines how much the images are scaled. Default is 0.15.

## Checking for Convexity

The script `analyze_space.py` can be used to check whether the categories within the space are convex. The script iterates over all categories, builds a convex hull of the items belonging to this category and counts how many points from other categories lie within this convex hull. Each point that lies in the convex hull of a different concept is counted as one violation. The script outputs the number of violations for each category, together with an estimate of how many violations would be expected if points are randomly sampled from a uniform distribution, a normal distribution, or the overall set of given points.

The script can be exectued as follows (where `n_dims` is the number of dimension of this specific MDS space):
```
python mds/analyze_space.py path/to/vectors.csv path/to/data.pickle n_dims
```

