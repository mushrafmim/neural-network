import numpy as np
from layer import Layer
import csv


class NN:

    def __init__(self, layers: list[Layer]) -> None:
        self.layers: list[Layer] = layers
        self.dWs = list()
        self.dbs = list()

        for i in range(1, len(self.layers)):
            self.layers[i].set_weights_and_biases(
                weights=self.get_weights_by_layer(i),
                biases=self.get_bias_by_layer(i)
            )

    def predict(self, X) -> np.array:

        output = X

        for i in range(1, len(self.layers)):
            output = self.layers[i].forward_pass(
                output)

        return output

    def find_error(self, Y_predicted, Y_i):
        cross_entropy_loss = np.dot(np.log(Y_predicted), Y_i)
        return cross_entropy_loss

    def train(self, X, Y):

        for i in range(len(X)):
            X_i = X[i]
            Y_i = Y[i]

            Y_i_pred = self.predict(X_i)

            error = self.find_error(Y_i_pred, Y_i)

            print(error)

    def get_weights_by_layer(self, layer_no: int):

        f_w = open('task_1/w.csv', 'rt')

        f_w_csv_reader = csv.reader(f_w)

        w_row_limits = [
            None,
            {'from_': 0, 'to': 14},
            {'from_': 14, 'to': 114},
            {'from_': 114, 'to': 155}
        ]

        w_out = []

        for row in f_w_csv_reader:

            if w_row_limits[layer_no]['to'] >= f_w_csv_reader.line_num > w_row_limits[layer_no]['from_']:
                w_out.append(row[1:])

        f_w.close()

        arr = np.array(w_out).astype(np.float64)

        return arr.transpose()

    def get_bias_by_layer(self, layer_no: int):

        f_b = open('task_1/b.csv', 'rt')

        f_b_csv_reader = csv.reader(f_b)

        b_out = []

        for row in f_b_csv_reader:

            if f_b_csv_reader.line_num == layer_no:
                b_out.append(row[1:])

        f_b.close()

        arr = np.array(b_out).astype(np.float64).transpose()

        return arr

        # return arr.transpose()

    def backpropogate(self, X, Y):

        Y_pred = self.predict(X)
        dZ = Y_pred - Y

        dW = self.layers[-2].find_dW(dZ)

        self.dWs.append(dW)
        self.dbs.append(dZ)

        for i in range(len(self.layers) - 2, 0, -1):
            relu_der = self.layers[i].relu_derivative()

            W = self.layers[i + 1].Weights.T

            dZ = (W.dot(dZ)) * relu_der

            dW = self.layers[i].find_dW(dZ)

            self.dWs.append(dW)
            self.dbs.append(dZ)

        self.dWs.reverse()
        self.dbs.reverse()

        self.write_dbs()
        self.write_dws()

    def write_dbs(self):
        with open('task_1/dbs.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(self.dbs)

    def write_dws(self):
        with open('task_1/dws.csv', 'w') as f:
            writer = csv.writer(f)

            for row in self.dWs:
                writer.writerows(row)
