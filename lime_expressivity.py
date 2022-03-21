from expressivity import extremize_costs_dataset
from lime.lime_tabular import LimeTabularExplainer
from linearexpressivity import compas_df
from sklearn.ensemble import RandomForestClassifier


def lime_cost_func_generator(classifier):
    def cost_func(dataset, feature_num):
        lime_exp = LimeTabularExplainer(dataset)

        def internal_cost(row):
            explanation = lime_exp.explain_instance(row, classifier.predict_proba, num_features=row.shape[0])
            return explanation.as_list()[feature_num][1]

        return internal_cost

    return cost_func


def split_out_dataset(dataset, target_column):
    x = dataset.drop(target_column, axis=1).to_numpy()
    y = dataset[target_column].to_numpy()
    return x, y


classifier = RandomForestClassifier()
x, y = split_out_dataset(compas_df, 'two_year_recid')
classifier.fit(x, y)

print(extremize_costs_dataset(compas_df, lime_cost_func_generator(classifier), target_column='two_year_recid'))
