import openxai
import torch
import torch.nn as nn
import numpy as np

class GradImpFunc:
    def __init__(self, classifier, X, y, seed):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.exps = []
        self.seed = seed

    # Populate importances
    # exps[n][i] returns expressivity of feature i in datapoint n
    def populate_exps(self):
        print("Not implemented here yet")
        exp_method = openxai.Explainer(method='grad', model=self.classifier, dataset_tensor=torch.from_numpy(self.X))
        explanations = exp_method.get_explanation(self.X, torch.from_numpy(self.y).long())
        explanation_values = explanations.numpy()
        for row in explanation_values:
            exp_dict = {}
            for i in range(len(row)):
                exp_dict[i] = np.float64(row[i])
            self.exps.append(exp_dict)
        pass

    # Given feature and row, return the computed expressivities
    def get_exp(self, row, feature):
        if len(self.exps) == 0:
            print("Expressivity dict empty. Populating now...")
            self.populate_exps()
        return self.exps[row][feature]

    def get_total_exp(self, assigns, feature_num):
        total_expressivity = 0
        for i in range(len(assigns)):
            total_expressivity += assigns[i] * self.exps[i][feature_num]
        return total_expressivity
