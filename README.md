# Model Explanation Disparities

This repo examines datasets to identify subgroups that are treated differently by a classification or regression model
compared to the whole dataset.

The pre-print describing the methodology in depth is available on [Arxiv](https://arxiv.org/abs/2303.01704).


## Locally Separable Important Notions

If the model explanation notion is separable in nature, use the following scripts. 
For clarification on notion separability, see Definition 2.2 in paper. LIME, SHAPly, and OpenXAI's Vanilla Gradient are
all examples of separable notions.

### Pre-processing Data (Recommended)

To enable repeat analysis, we recommend pre-processing the importance values that your explanation model provides.
Using process_exps.py, input your dataset, classifier/regression model, and model explanation method.
LIME, SHAP, and Gradient are already implemented.

The resulting data will be output in a json format that can be read in during the initial phase of the optimization.

Importantly, you must set a randomized seed for both your classifier and your model explanation module in order to replicate
the results from experiment to experiment.

### Running Constrained Optimization Algorithm

In the constrained_opt.py script, input the dataset of interest, specifying the target feature, train/test split,
and sensitive features. You may also adjust the desired $\alpha$ range. Run the script using:

'''
python constrained_opy.py <importance notion>
'''

## Non-Separable Case

linearexpressivity.py contains the primary code used for this. Use run_linear.py to specify parameters
such as dataset, target feature, sensitive features, and alpha range and to run the algorithm.

The optional flag --cuda can be used to enable gpu processing. This is highly recommended for speed purposes.


#### Note on flat_val

flat_val is the regularization constant to prevent exploding gradients in the optimization
process. For smaller datasets ($<10k$ data points), this value can be set around $.0001$. For larger
datasets, this value may need to be larger.


## Datasets

Our initial analysis was performed on four datasets: COMPAS, Student, Bank, and Folktables (MI Income).
See datasets.ipynb for descriptions of the datasets and links for downloading the data.
