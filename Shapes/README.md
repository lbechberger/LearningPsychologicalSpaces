# Shapes

## About

Our scripts use TensorFlow 1.4.1 with Python 3.5 along with scikit-learn. You can find scripts for setting up a virtual environment with anaconda in the [Utilities](https://github.com/lbechberger/Utilities) project. 

Please find more detailed instructions on how to use the scripts below in the respective sections of the workflow.

## Multidimensional Scaling

### Preprocessing

In order to make the original data processible by our scripts, please run the script `preprocess.py` as follows:
```
python preprocess.py path/to/within.csv path/to/within_between.csv path/to/output.pickle
```

The file `within.csv` contains within category similarity judments, the file `within_between.csv` contains similarity ratings both within and between categories. All of these similarity ratings are based on shape similarity only (participants were explicitly asked to not consider *conceptual* similarity).

The resulting `output.pickle` file contains a dictionary with the follwing elements:
- 'categories': A dictionary using category names as keys and containing dictionaries as values. These dictionaries have the following elements:
..- 'visSim': Is the category visually homogeneous? 'Sim' means homogeneous, 'Dis' means not homogeneous, and 'x' means unclear.
..- 'artificial': Does the category consist of natural ('nat') or artificial ('art')?
..- 'items': A list of the IDs of all items that belong into this category. 
- 'items': A dictionary using item IDs as keys and containing dictionaries as values. These dictionaries have the following elements:
..- 'name': A human readable name for the item.
..- 'category': The name of the category this item belongs to.
- 'similarities': A dictionary using the string representation of sets of two items as keys and dictionaries as values. These dictionaries have the following elements:
..- 'relation': Is this a 'within' category or a 'between' category rating?
..- 'values': A list of similarity values (integers from 1 to 5, where 1 means 'no visual similarity' and 5 means 'high visual similarity')
..- 'border': An integer indicating the border between similarity ratings from the two studies. You can use `values[:border]` to access only the similarity ratings of the first study (only within category) and `values[border:]` to acces only the similarity ratings of the second study (mostly between cateory, but also some within category).
