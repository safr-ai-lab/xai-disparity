from reg_oracle import ZeroPredictor, CostPredictor, RegOracle



def fit_one_side(features, costs, minimize = False):
    left_predictor = CostPredictor()
    right_predictor = ZeroPredictor()
    left_predictor.fit(features, costs)
    reg_oracle = RegOracle(left_predictor, right_predictor, minimize=minimize)
    return reg_oracle
