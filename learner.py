import copy
from sklearn import linear_model
from reg_oracle import RegOracle
# This class sourced from the gerryfair repo

class Learner:
    def __init__(self, X, y, predictor):
        self.X = X
        self.y = y
        self.predictor = predictor

    def best_response(self, costs_0, costs_1, minimize=True):
        """Solve the CSC problem for the learner."""
        reg0 = copy.deepcopy(self.predictor)
        reg0.fit(self.X, costs_0)
        reg1 = copy.deepcopy(self.predictor)
        reg1.fit(self.X, costs_1)
        func = RegOracle(reg0, reg1, minimize)
        return func
