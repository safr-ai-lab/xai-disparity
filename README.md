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
Using process_exps.py, input your dataset, classifier/regression model, and model explanation method (must be implemented
in a class similar to that available to those already in notions directory). LIME, SHAP, and Gradient are already implemented.

The resulting data will be output in a json format that can be read in during the initial phase of the optimization.

Importantly, you must set a randomized seed for both your classifier and your model explanation module in order to replicate
the results from experiment to experiment.

### Running Constrained Optimization Algorithm

The main code for the script is available in constrained_opt.py script. Use run_separable.py to specify parameters
such as dataset, target feature, sensitive features, importance notion, and alpha range and to run the algorithm.

#### Note on hyperparameters

Depending on your use case, you may need to modify hyperparameters to ensure smooth convergence. Within constrained_opt.py,
argmin_g can be modified to fit your uses. $v$ controls the error tolerance of the algorithm.

The maximum iterations limit can also be set to prevent degenerate cases.


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
