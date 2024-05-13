# Model Explanation Disparities

This repo evaluates prediction models to identify subgroups of the dataset that have
Feature Importance Disparities (FID). This is when the importance of a certain feature, as measured
by a chosen importance metric such as LIME or SHAP, is substantially different on the subgroup versus on the
whole dataset. While testing this for subgroups identified by a singular sensitive characteristic
is a straightforward problem, we search the exponentially large space of "rich" subgroups, which can
be defined by a function of the sensitive characteristics.

For further breakdown of the methodology and background, see the pre-print on [Arxiv](https://arxiv.org/abs/2303.01704).

This repo contains code to:

- Identify high FID subgroups using "separable" importance notions (such as LIME).
- Identify high FID subgroups using "nonseparable" importance notions (such as linear regression coefficients).
- Compute feature importance values for the LIME, SHAP, and Gradient notions.
- Run various additional experiments from the paper such as searching the marginal subgroup space for high FID subgroups.


#### Prerequisites

To clone and run, do:

```
git clone https://github.com/safr-ml-lab/xai-disparity.git
pip install -r requirements.txt
```

#### A Note on Datasets

All feature variables must be either continuous or binary. If you have categorical variables, we recommend one-hot
encoding.

If you are using Vanilla Gradient as your importance notion, your dataset must have a binary target column.


## Locally Separable Important Notions

If the model explanation notion is separable in nature, use the following scripts. 
For clarification on separability, see Definition 3.2 in paper. LIME, SHAPly, and OpenXAI's Vanilla Gradient are
all examples of separable notions.

### Pre-processing Data (Recommended)

To enable repeat analysis, we highly recommend pre-processing the importance values that your explanation model provides.
This can be done using the process_imps.py script. LIME, SHAP, and Vanilla Gradient are already implemented and can be specified
during execution. If you are using a different importance notion, you must implement a new module (see notions/ for examples).

Within process_imps.py, specify your dataset and also input the classifier/regression model being used. Run using:

```
python process_imps.py <importance notion>
```

The resulting data will be output in a json format that can be read in during the initial phase of the optimization.
This json file should be placed in input/imps/

Importantly, you must set a randomized seed for both your classifier and your model explanation module in order to replicate
the results from experiment to experiment.

### Running Constrained Optimization Algorithm

The main underlying codebase is in the constrained_opt.py script and is run using the run_separable.py script.
In run_separable.py, you specify the details including dataset, classifier, alpha range, and parameters.  
Run the code using:

```
python run_separable.py --method <importance notion> --file [importance file]
```

If you pre-processed your importances, specify the [importance file] name without the "train/test" string. 
i.e. the output of process_imps.py could be "student_lime_seed0_train" and "student_lime_seed0_test"
but the command input would be just "student_lime_seed0".

If you did not pre-process your importances, run_separable.py will compute them at run time to use. It will also
save the computed importances to a json file in input/imps/


#### Note on hyperparameters

Depending on your use case, you may need to modify hyperparameters to ensure smooth convergence. Within constrained_opt.py,
argmin_g can be modified to fit your uses. $v$ controls the error tolerance of the algorithm. 
The maximum iterations limit can also be set to prevent degenerate cases.


## Non-Separable Case

linearexpressivity.py contains the primary underlying code used for finding subgroups with high FID using Linear Regression
as the feature importance notion. In the script run_linear.py, specify parameters
such as dataset, target feature, sensitive features, and alpha range. Run the code using:

```
python run_linear.py
```

The optional flag --cuda can be used to enable gpu processing. This is highly recommended for speed purposes.


#### Note on flat_val

flat_val is the regularization constant to prevent exploding gradients in the optimization
process. For smaller datasets ($<10k$ data points), this value can be set around $.0001$. For larger
datasets, this value may need to be larger.


## Datasets

Our experiments were performed on four datasets:

- Student: predicting student final math grades using personal and academic records.
- COMPAS: predicting recidivism risk for pre-trail criminal defendants
- Bank: predicting whether an individual would sign up for a bank account
- Folktables (ACSIncome): predicting whether an individual makes more than 50,000 dollars.

See [input/datasets.ipynb](https://github.com/safr-ml-lab/xai-disparity/blob/master/input/datasets.ipynb)
for further descriptions of the datasets and links for downloading the data.


## Contact

- Repo maintained by Peter Chang <pchang@hbs.edu> <pchang534@gmail.com> 
- Property of SAFR AI Lab, $D^3$ Institute, Harvard Business School. PI: Seth Neel <sneel@hbs.edu>
- Contributors: Peter Chang, Leor Fishman <leor.fishman@gmail.com>, and Seth Neel
