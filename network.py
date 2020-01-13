import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


class LstmCell:
    # h: number of hidden units, d: number of features in input
    def __init__(self, prev_ct, prev_ht, h, d):
        # Vectors
        self.input = np.zeros(d)
        self.ft = np.zeros(h)
        self.it = np.zeros(h)
        self.ot = np.zeros(h)
        self.ct_bar = np.zeros(h)
        self.ht = np.zeros(h)
        self.ct = np.zeros(h)

        self.prev_ct = prev_ct
        self.prev_ht = prev_ht
        # Weights for input vector
        self.Wf = np.random.randn(h, d)
        self.Wi = np.random.randn(h, d)
        self.Wo = np.random.randn(h, d)
        self.Wc = np.random.randn(h, d)
        # Weights for hidden state vector
        self.Uf = np.random.randn(h, h)
        self.Ui = np.random.randn(h, h)
        self.Uo = np.random.randn(h, h)
        self.Uc = np.random.randn(h, h)
        # Bias vectors
        self.bf = np.random.randn(h)
        self.bi = np.random.randn(h)
        self.bo = np.random.randn(h)
        self.bc = np.random.randn(h)

    def feedforward(self, input):
        self.input = input
        self.ft = sigmoid(self.Wf@input + self.Uf@self.prev_ht + self.bf)  # forget gate
        self.it = sigmoid(self.Wi@input + self.Ui@self.prev_ht + self.bi)  # update gate
        self.ot = sigmoid(self.Wo@input + self.Uo@self.prev_ht + self.bo)  # output gate
        self.ct_bar = tanh(self.Wc @ input + self.Uc @ self.prev_ht + self.bc)
        # outputs
        self.ct = np.multiply(self.ft, self.prev_ct) + np.multiply(self.it, self.ct_bar)
        self.ht = np.multiply(self.ot, tanh(self.ct))
        return self.ct, self.ht

    def backpropagate(self, ct_error, ht_error):
        # Weight and bias deltas, counted in groups (Wk, Uk, bk) where k goes through (f, i, o, c)
        # First I count the ht_error derivative and then the ct_error derivative (adding it at the same time)

        # (firstly) ht_error with respect to Wf
        ht_ct = self.ot * tanh_deriv(self.ct)
        ct_ft = self.prev_ct
        ft_Sf = sigmoid_deriv(self.Wf@input + self.Uf@self.prev_ht + self.bf)
        Wf_deltas = np.dot((ht_error * ht_ct * ct_ft * ft_Sf).reshape((-1, 1)), self.input)
        # add ct_error with respect to Wf to Wf_deltas
        Wf_deltas += np.dot((ct_error * ct_ft * ft_Sf).reshape((-1, 1)), self.input)
        # ht_error with respect to Uf
        Uf_deltas = np.dot((ht_error * ht_ct * ct_ft * ft_Sf).reshape((-1, 1)), self.prev_ht)
        # add ct_error with respect to Uf
        Uf_deltas += np.dot((ct_error * ct_ft * ft_Sf).reshape((-1, 1)), self.prev_ht)
        # ht_error with respect to bf (forget gate bias)
        bf_deltas = ht_error * ht_ct * ct_ft * ft_Sf
        # ct_error with respect to bf added to bf_deltas
        bf_deltas += ct_error * ct_ft * ft_Sf

        # ht_error with respect Wi
        ct_it = self.ct_bar
        it_Si = sigmoid_deriv(self.Wi@input + self.Ui@self.prev_ht + self.bi)
        Wi_deltas = np.dot((ht_error * ht_ct * ct_it * it_Si).reshape((-1, 1)), self.input)
        # ct_error with respect to Wi
        Wi_deltas += np.dot((ct_error * ct_it * it_Si).reshape((-1, 1)), self.input)
        # ht_error with respect to Ui
        Ui_deltas = np.dot((ht_error * ht_ct * ct_it * it_Si).reshape((-1, 1)), self.prev_ht)
        # ct_error with respect to Ui
        Ui_deltas += np.dot((ct_error * ct_it * it_Si).reshape((-1, 1)), self.prev_ht)
        # update gate bias deltas
        bi_deltas = ht_error * ht_ct * ct_it * it_Si + ct_error * ct_it * it_Si

        # ht_error with respect to Wo
        ht_ot = tanh(self.ct)
        ot_So = sigmoid_deriv(self.Wo@input + self.Uo@self.prev_ht + self.bo)
        Wo_deltas = np.dot((ht_error * ht_ot * ot_So).reshape((-1, 1)), self.input)
        # There is no cell state error with respect to ot
        # ht_error with respect to Uo
        Uo_deltas = np.dot((ht_error * ht_ot * ot_So).reshape((-1, 1)), self.prev_ht)
        # ouput gate bias deltas
        bo_deltas = ht_error * ht_ot * ot_So

        # ht_error with respect to Wc
        ct_ctbar = self.it
        ctbar_Sc = tanh_deriv(self.Wc @ input + self.Uc @ self.prev_ht + self.bc)
        Wc_deltas = np.dot((ht_error * ht_ct * ct_ctbar * ctbar_Sc).reshape((-1, 1)), self.input)
        # ct_error with respect to Wc
        Wc_deltas += np.dot((ct_error * ct_ctbar * ctbar_Sc).reshape((-1, 1)), self.input)
        # ht_error with respect to Uc
        Uc_deltas = np.dot((ht_error * ht_ct * ct_ctbar * ctbar_Sc).reshape((-1, 1)), self.prev_ht)
        # ct_error with respect to Uc
        Uc_deltas += np.dot((ct_error * ct_ctbar * ctbar_Sc).reshape((-1, 1)), self.prev_ht)
        # cell gate bias deltas
        bc_deltas = ht_error * ht_ct * ct_ctbar * ctbar_Sc + ct_error * ct_ctbar * ctbar_Sc

        # Then we must differentiate the errors with respect to previous cell's hidden state and cell state for it to be
        # able to continue backpropagation

        # ht_error with respect to prev_ht
        ft_prevht = sigmoid_deriv(self.Wf@input + self.Uf@self.prev_ht + self.bf)@self.Uf
        it_prevht = sigmoid_deriv(self.Wi@input + self.Ui@self.prev_ht + self.bi)@self.Ui
        ctbar_prevht = tanh_deriv(self.Wc @ input + self.Uc @ self.prev_ht + self.bc)@self.Uc

        ct_prevht = self.prev_ct * ft_prevht + it_prevht * self.ct_bar + ctbar_prevht * self.it

        ht_prevht = sigmoid_deriv(self.Wo@input + self.Uo@self.prev_ht + self.bo)@self.Uo * tanh(self.ct) + ht_ct * ct_prevht

        new_ht_error = ht_error * ht_prevht

        # ct_error with respect to prev_ht
        new_ht_error += ct_prevht

        # ht_error with respect to prev_ct
        new_ct_error = ht_ct * self.ft
        # ct_error with respect to prev_ct
        new_ct_error += self.ft
