import torch
from torch import Tensor
import practical_prologue as prologue 


def sigma(x):
    return torch.tanh(x)


def dsigma(x): 
    return 1 - torch.pow(torch.tanh(x),2)


def loss(v, t): 
    return torch.sum(torch.pow((t - v), 2))


def forward_pass(w1, b1, w2, b2, x): 
    s1 = torch.mv(w1, x) + b1
    x1 = sigma(s1)

    s2 = torch.mv(w2, x1) + b2
    x2 = sigma(s2)

    return x, s1, x1, s2, x2


def backward_pass(w1, b1, w2, b2,
                  t, 
                  x, s1, x1, s2, x2,
                  dl_dw1, dl_db1, dl_dw2, dl_db2):
    # network params 
    # t is target vector 
    # quantities computed in fwd pass 
    # tensors used to store cumulated sums of gradient on individual samples 
    # updates latters based on formula of backward pass 

    dl_dx2 = -2*(t-x2)
    dl_ds2 = dl_dx2 * dsigma(s2)
    dl_dw2_i = torch.mm(dl_ds2.view(dl_ds2.shape[0], 1), x1.view(1, x1.shape[0]))

    dl_dx1 = torch.mm(w2.t(), dl_ds2.view(dl_ds2.shape[0], 1))
    dl_dx1 = dl_dx1.view(dl_dx1.shape[0])
    dl_ds1 = dl_dx1 * dsigma(s1)
    dl_dw1_i = torch.mm(dl_ds1.view(dl_ds1.shape[0], 1), x.view(1, x.shape[0]))

    dl_db1_i = dl_ds1 
    dl_db2_i = dl_ds2

    # update 
    dl_dw1 += dl_dw1_i
    dl_dw2 += dl_dw2_i
    dl_db1 += dl_db1_i
    dl_db2 += dl_db2_i

    return dl_dw1, dl_db1, dl_dw2, dl_db2 
    

def step(x, t,
        w1, b1, w2, b2, 
        dl_dw1, dl_db1, dl_dw2, dl_db2): 
    # forward pass 
    x0, s1, x1, s2, x2 = forward_pass(
            w1, b1, w2, b2, x)

    # backward pass
    dl_dw1, dl_db1, dl_dw2, dl_db2 = backward_pass(
                    w1, b1, w2, b2,
                    t, 
                    x0, s1, x1, s2, x2, 
                    dl_dw1, dl_db1, dl_dw2, dl_db2
                    )
    
    return dl_dw1, dl_dw2, dl_db1, dl_db2


def predict_score(x_test, t_test, w1, b1, w2, b2):
    predictions = torch.empty(len(x_test),n_layer_2)
    for i in range(len(x_test)):
        _, _, _, _, pred = forward_pass(w1, b1, w2, b2, x_test[i])

    pred_index = torch.argmax(predictions, dim=-1)

    # compare to true value 
    test_index = torch.argmax(t_test, dim=-1)

    # get number of nonzero (incorrect)
    incorrect = torch.count_nonzero(test_index - pred_index)
    # percentage correct 
    score = (len(x_test) - incorrect) / len(x_test) 
    return score


if __name__ == "__main__":
    # load data 
    x, t, x_test, t_test = prologue.load_data(
            one_hot_labels=True,
            normalize=True
            )

    # multiply target label vectors by 0.9 to make sure theyre in the range of tanh 
    t = t * 0.9
    t_test = t_test * 0.9

    n_layer_1 = 50
    n_layer_2 = 10
    step_size = 0.1 / t.shape[0]

    # create four weight and bias tensors 
    # fill with random values samples from normal distribution N(0, 1e-6)
    w1 = torch.empty(n_layer_1, x.shape[-1]).normal_(mean=0, std=1e-6)
    b1 = torch.empty(n_layer_1).normal_(mean=0, std=1e-6) 
    w2 = torch.empty(n_layer_2, n_layer_1).normal_(mean=0, std=1e-6)
    b2 = torch.empty(n_layer_2).normal_(mean=0, std=1e-6)

    for s in range(10):
        print('iter', s)
        # create tensors to sum up gradients 
        dl_dw1 = torch.zeros(w1.shape)
        dl_db1 = torch.zeros(b1.shape)
        dl_dw2 = torch.zeros(w2.shape)
        dl_db2 = torch.zeros(b2.shape)

        # update derivatives for all training data
        for i in range(len(x)):
            dl_dw1, dl_dw2, dl_db1, dl_db2 = step(
                x[i], t[i], 
                w1, b1, w2, b2, 
                dl_dw1, dl_db1, dl_dw2, dl_db2
                )

        # update params for average derivatives over training data
        w1 -= step_size * dl_dw1
        w2 -= step_size * dl_dw2
        b1 -= step_size * dl_db1
        b2 -= step_size * dl_db2

        train_score = predict_score(x, t, w1, b1, w2, b2)
        print('train score', train_score)
        test_score = predict_score(x_test, t_test, w1, b1, w2, b2)
        print('test score', test_score)




