import openxai
import torch
import torch.nn as nn
import numpy as np

class GradImpFunc:
    def __init__(self, classifier, X, y, seed):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.imps = []
        self.seed = seed

    # Populate importances
    # imps[n][i] returns importance of feature i in datapoint n
    def populate_imps(self):
        print("Not implemented here yet")
        imp_method = openxai.Explainer(method='grad', model=self.classifier, dataset_tensor=torch.from_numpy(self.X))
        explanations = imp_method.get_explanation(self.X, torch.from_numpy(self.y).long())
        explanation_values = explanations.numpy()
        for row in explanation_values:
            imp_dict = {}
            for i in range(len(row)):
                imp_dict[i] = np.float64(row[i])
            self.imps.append(imp_dict)
        pass

    # Given feature and row, return the computed importance
    def get_imp(self, row, feature):
        if len(self.imps) == 0:
            print("Importance dict empty. Populating now...")
            self.populate_imps()
        return self.imps[row][feature]

    def get_total_imp(self, assigns, feature_num):
        total_importance = 0
        for i in range(len(assigns)):
            total_importance += assigns[i] * self.imps[i][feature_num]
        return total_importance
