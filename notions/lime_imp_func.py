from lime.lime_tabular import LimeTabularExplainer
import re

class LimeImpFunc:
    def __init__(self, classifier, X, y, seed):
        self.classifier = classifier
        self.X = X
        self.imps = []
        self.lime_exp = LimeTabularExplainer(self.X, random_state=seed)

    # Populate imps with importance dictionaries
    # imps[n][i] returns importance of feature i in datapoint n
    def populate_imps(self):
        i = 0
        for row in self.X:
            if i % 100 == 0:
                print(i, '/', len(self.X))
            #print('Computing ', i)
            imp_i = self.lime_exp.explain_instance(row, self.classifier.predict_proba, num_features=row.shape[0]).as_list()
            imp_dict = {}
            # Clean up LIME output and return dict with key=feature, value=expressivity
            for e in imp_i:
                parts = re.split(r"[\<=\<\>=\>]", e[0].replace(" ", ""))
                for p in parts:
                    if '.' not in p and len(p)>0:
                        feature = int(p)
                imp_dict[feature] = e[1]
            self.imps.append(imp_dict)
            i += 1

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
