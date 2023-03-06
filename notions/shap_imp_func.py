import shap

class ShapImpFunc:
    def __init__(self, classifier, X, y, seed):
        self.classifier = classifier
        self.X = X
        self.y = y
        self.imps = []
        self.explainer = shap.Explainer(classifier.predict, X, seed=seed)

    # Populate imps with importance dictionaries
    # imps[n][i] returns importance of feature i in datapoint n
    def populate_imps(self):
        shap_values = self.explainer(self.X)
        for row in shap_values.values:
            imp_dict = {}
            for i in range(len(row)):
                imp_dict[i] = row[i]
            self.imps.append(imp_dict)

    # Given feature and row, return the computed importance
    def get_imp(self, row, feature):
        if len(self.imps) == 0:
            print("importance dict empty. Populating now...")
            self.populate_imps()
        return self.imps[row][feature]

    def get_total_imp(self, assigns, feature_num):
        total_importance = 0
        for i in range(len(assigns)):
            total_importance += assigns[i] * self.imps[i][feature_num]
        return total_importance
