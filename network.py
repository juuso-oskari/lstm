
import cupy as cp


def tanh(x):
    return cp.tanh(x)


def tanh_deriv(x):
    return 1.0 - cp.tanh(x) ** 2


def sigmoid(x):
    return cp.array(1 / (1 + cp.exp(-x)))


def sigmoid_deriv(x):
    return cp.array(sigmoid(x) * (1 - sigmoid(x)))

def error_deriv(target, prediction):
    return target - prediction


class LstmCell:
    # h: number of hidden units, d: number of features in input
    def __init__(self, prev_ct, prev_ht, h, d):
        # Vectors
        self.input = cp.zeros(d)
        self.ft = cp.zeros(h)
        self.it = cp.zeros(h)
        self.ot = cp.zeros(h)
        self.ct_bar = cp.zeros(h)
        self.ht = cp.zeros(h)
        self.ct = cp.zeros(h)

        self.prev_ct = prev_ct
        self.prev_ht = prev_ht
        # Weights for input vector
        self.Wf = cp.random.randn(h, d)
        self.Wi = cp.random.randn(h, d)
        self.Wo = cp.random.randn(h, d)
        self.Wc = cp.random.randn(h, d)
        # Weights for hidden state vector
        self.Uf = cp.random.randn(h, h)
        self.Ui = cp.random.randn(h, h)
        self.Uo = cp.random.randn(h, h)
        self.Uc = cp.random.randn(h, h)
        # Bias vectors
        self.bf = cp.random.randn(h)
        self.bi = cp.random.randn(h)
        self.bo = cp.random.randn(h)
        self.bc = cp.random.randn(h)

    def feedforward(self, input):
        self.input = input
        self.ft = sigmoid(self.Wf@input + self.Uf@self.prev_ht + self.bf)  # forget gate
        self.it = sigmoid(self.Wi@input + self.Ui@self.prev_ht + self.bi)  # update gate
        self.ot = sigmoid(self.Wo@input + self.Uo@self.prev_ht + self.bo)  # output gate
        self.ct_bar = tanh(self.Wc @ input + self.Uc @ self.prev_ht + self.bc)
        # outputs
        self.ct = cp.multiply(self.ft, self.prev_ct) + cp.multiply(self.it, self.ct_bar)
        self.ht = cp.multiply(self.ot, tanh(self.ct))
        return self.ct, self.ht

    def backpropagate(self, ct_error, ht_error):

        # part of the ht_error propagates back to cell state and to other gates besides the output gate
        cell_error = ct_error + ht_error * self.ot * tanh_deriv(self.ct)

        # ot (output gate only affects ht_error)
        d_ot = ht_error * tanh(self.ct) * sigmoid_deriv(self.Wo@input + self.Uo@self.prev_ht + self.bo)

        d_Wo = cp.matmul(cp.reshape(d_ot, (-1, 1)), cp.reshape(self.input, (1, -1)))
        d_Uo = cp.matmul(cp.reshape(d_ot, (-1, 1)), cp.reshape(self.prev_ht, (1, -1)))
        d_bo = d_ot

        # cbar
        d_cbar = cell_error * self.it * tanh_deriv(self.Wc @ input + self.Uc @ self.prev_ht + self.bc)

        d_Wc = cp.matmul(cp.reshape(d_cbar, (-1, 1)), cp.reshape(self.input, (1, -1)))
        d_Uc = cp.matmul(cp.reshape(d_cbar, (-1, 1)), cp.reshape(self.prev_ht, (1, -1)))
        d_bc = d_cbar

        # it
        d_it = cell_error * self.ct_bar * sigmoid_deriv(self.Wi@input + self.Ui@self.prev_ht + self.bi)

        d_Wi = cp.matmul(cp.reshape(d_it, (-1, 1)), cp.reshape(self.input, (1, -1)))
        d_Ui = cp.matmul(cp.reshape(d_it, (-1, 1)), cp.reshape(self.prev_ht, (1, -1)))
        d_bi = d_it

        # ft
        d_ft = cell_error * self.prev_ct * sigmoid_deriv(self.Wf@input + self.Uf@self.prev_ht + self.bf)

        d_Wf = cp.matmul(cp.reshape(d_ft, (-1, 1)), cp.reshape(self.input, (1, -1)))
        d_Uf = cp.matmul(cp.reshape(d_ft, (-1, 1)), cp.reshape(self.prev_ht, (1, -1)))
        d_bf = d_ft

        # to continue backpropagation in the previous LSTM-cell,
        # we must backpropagate errors to prev_ht and prev_ct

        # prev_ht_error
        # We have to consider the lower route, strictly from ht_error (=d_ot), and upper route,
        # adding the gate errors (thus adding all gate errors together)

        prev_ht_error = d_ot + d_cbar + d_it + d_ft

        # prev_ct_error
        # it is pretty straightforward, we have to only concern about cell_error's modulation part with ft-gate

        prev_ct_error = cell_error * self.ft

        # for return, lets create dictionaries for errors and deltas
        deltas = dict(Wo=d_Wo, Uo=d_Uo, bo=d_bo, Wc=d_Wc, Uc=d_Uc, bc=d_bc, Wi=d_Wi,
                      Ui=d_Ui, bi=d_bi, Wf=d_Wf, Uf=d_Uf, bf=d_bf)
        errors = (prev_ct_error, prev_ht_error)

        return errors, deltas

    def updateParameters(self, deltas, eta):
        self.Wf -= eta*deltas["Wf"]
        self.Wi -= eta*deltas["Wi"]
        self.Wo -= eta*deltas["Wo"]
        self.Wc -= eta*deltas["Wc"]
        # Weights for hidden state vector
        self.Uf -= eta*deltas["Uf"]
        self.Ui -= eta*deltas["Ui"]
        self.Uo -= eta*deltas["Uo"]
        self.Uc -= eta*deltas["Uc"]
        # Bias vectors
        self.bf -= eta*deltas["bf"]
        self.bi -= eta*deltas["bi"]
        self.bo -= eta*deltas["bo"]
        self.bc -= eta*deltas["bc"]





class LstmNetwork:
    def __init__(self, h, d, sequence_length):

        self.h, self.d = h, d
        self.sequence_length = sequence_length
        self.cells = [LstmCell(cp.zeros(h), cp.zeros(h), h, d) for i in range(sequence_length)]
        self.deltas = dict()

    def feedforward(self, sequence):

        if len(sequence) != self.sequence_length:
            raise Exception('input sequence length: {} did not match the lstm-cell number : {}'
                            .format(len(sequence), self.sequence_length))
        else:

            # in the first cell, prev_ct and prev_ht are zero-vectors
            prev_ct, prev_ht = cp.zeros(self.h), cp.zeros(self.h)
            for i in range(self.sequence_length):
                self.cells[i].prev_ct = prev_ct
                self.cells[i].prev_ht = prev_ht
                prev_ct, prev_ht = self.cells[i].feedforward(sequence[i])

            prediction = prev_ht

        return prediction

    def backpropagate(self, targets, predictions, eta):

        prediction_error = cp.zeros(self.h)
        n = len(targets)

        for target, prediction in zip(targets, predictions):
            prediction_error = error_deriv(target, prediction)

            ct_error, ht_error = cp.zeros(self.h), prediction_error

            for i in range(1, self.sequence_length + 1):
                new_errors, deltas = self.cells[-i].backpropagate(ct_error, ht_error)
                self.cells[-i].updateParameters(deltas, 1/n*eta)    # this might be a slow way
                ct_error, ht_error = new_errors






if __name__ == "__main__":

    lstm_cell_1 = LstmCell(cp.array([0]), cp.array([0]), 1, 2)
    ct_1, ht_1 = lstm_cell_1.feedforward(cp.array([1, 2]))

    lstm_cell_2 = LstmCell(ct_1, ht_1, 1, 2)
    ct_2, ht_2 = lstm_cell_2.feedforward(cp.array([0.5, 3]))

    target_1 = cp.array([0.5])
    target_2 = cp.array([1.25])

    ht_error = ht_2 - target_2

    ht_error_1 = ht_1 - target_1

    ct_error, ht_error, deltas_2 = lstm_cell_2.backpropagate(0, ht_error)

    ct_error, ht_error, deltas_1 = lstm_cell_1.backpropagate(ct_error, ht_error)

    ct_error, ht_error, deltas_1_add = lstm_cell_1.backpropagate(0, ht_error_1)

    print(deltas_2["Wf"])
    print(deltas_2["Wi"])
    print(deltas_2["Wo"])
    print(deltas_2["Wc"])

    print(cp.matmul(cp.array([])))




