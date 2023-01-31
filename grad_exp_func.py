import openxai
import torch
import torch.nn as nn

class GradExpFunc:
    def __init__(self, classifier, dataset, seed):
        self.classifier = classifier
        self.dataset = dataset
        self.exps = []
        #self.explainer = shap.Explainer(classifier.predict, dataset, seed=seed)

    # Populate exps with expressivity dictionaries
    # exps[n][i] returns expressivity of feature i in datapoint n
    def populate_exps(self):
        print("Not implemented here yet")
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
