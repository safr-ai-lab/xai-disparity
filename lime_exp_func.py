from lime.lime_tabular import LimeTabularExplainer
import re


class LimeExpFunc:
    def __init__(self, classifier, dataset, seed):
        self.classifier = classifier
        self.dataset = dataset
        self.exps = []
        self.lime_exp = LimeTabularExplainer(self.dataset, random_state=seed)

    # Populate exps with expressivity dictionaries
    # exps[n][i] returns expressivity of feature i in datapoint n
    def populate_exps(self):
        i = 0
        for row in self.dataset:
            #print('Computing ', i)
            exp_i = self.lime_exp.explain_instance(row, self.classifier.predict_proba, num_features=row.shape[0]).as_list()
            exp_dict = {}
            # Clean up LIME output and return dict with key=feature, value=expressivity
            for e in exp_i:
                parts = re.split(r"[\<=\<\>=\>]", e[0].replace(" ", ""))
                for p in parts:
                    if '.' not in p and len(p)>0:
                        feature = int(p)
                exp_dict[feature] = e[1]
            self.exps.append(exp_dict)
            i += 1

    # Given feature and row, return the computed expressivities
    def get_exp(self, row, feature):
        if len(self.exps) == 0:
            print("Expressivity dict empty. Populating now...")
            self.populate_exps()
        return self.exps[row][feature]
