from sklearn.linear_model import LinearRegression
# This class sourced from the gerryfair repo
class RegOracle:
    """Class RegOracle, linear threshold classifier."""

    def __init__(self, b0, b1, minimize=False):
        """

        :param b0: an oracle for the cost of assigning 0
        :param b1: an oracle for the cost of assigning 1
        :param minimize: whether to minimize or maximize cost
        """
        self.b0 = b0
        self.b1 = b1
        self.minimize = minimize

    def predict(self, x):
        """Predict labels on data set x."""
        reg0 = self.b0
        reg1 = self.b1
        n = x.shape[0]
        y = []
        total_cost = 0
        for i in range(n):
            c_0 = reg0.predict([x[i][:]])
            c_1 = reg1.predict([x[i][:]])
            y_i = int(c_1 < c_0)
            cost = min(c_0, c_1)
            if not self.minimize:
                y_i = 1 - y_i
                cost = max(c_0, c_1)
            y.append(y_i)
            total_cost += cost[0]
        return y, total_cost


class ZeroPredictor:
    """
    An 'oracle' that always predicts a cost of 0
    """
    @staticmethod
    def predict(x):
        """
        returns a vector of all 0 predictions
        """
        return [0 for _ in range(len(x))]

    @staticmethod
    def fit(_, __):
        """
        a blank training loop, needed for duck typing of the predictor
        :param _: ignored
        :param __: ignored
        :return:
        """
        return


ExpPredictor = LinearRegression
