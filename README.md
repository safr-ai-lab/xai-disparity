# Differential Expressivity

See datasets.ipynb for descriptions of the datasets that we are evaluating based on: COMPAS, Student, Bank, and Folktables (MI Income).
This notebook also includes links for downloading the datasets.

## Locally Separable Case

Currently implemented using LIME as the feature expressivity value. Run using:

"'
python local_sep_expressivity.py
"'

Include the flag --dummy if you want to scramble the y values for comparison purposes.

## Non-Separable Case

Uses linear regression to define feature expressivity. Run using:

"'
python linearexpressivity.py
"'

Include the flag --dummy if you want to scramble the y values for comparison purposes.

