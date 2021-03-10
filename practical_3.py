import torch
from torch import Tensor
import practical_prologue as prologue 


def sigma(x):
    return torch.tanh(x)


def dsigma(x): 
    # apply first derivative of tanh 
    # derivative of tanh = (cosh^2 - sinh^2)/cosh^2
    cosh2 = torch.pow(torch.cosh(x), 2)
    sinh2 = torch.pow(torch.sinh(x), 2)
    return (cosh2 - sinh2) / cosh2


def loss(v, t): 
    # v is the predicted tensor 
    # t is the target one 
    # return l2-norm squared i.e. sum of the squares
    return torch.sum(torch.pow((t - v), 2))


def forward_pass(w1, b1, w2, b2, x): 
    # input vector x 
    # weight and bias of 2 layers 
    # returns tuple x0, s1, x1, s2, x2
    x0 = x
    # broadcasting 
#    b1 = b1.view([b1.shape[0], 1])
    s1 = torch.mv(w1, x) + b1
    x1 = sigma(s1)

 #   b2 = b2.view([b2.shape[0], 1])
    s2 = torch.mv(w2, x1) + b2
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2


def backward_pass(w1, b1, w2, b2,
                  t, 
                  x, s1, x1, s2, x2,
                  dl_dw1, dl_db1, dl_dw2, dl_db2):
    # network params 
    # t is target vector 
    # quantities computed in fwd pass 
    # tensors used to store cumulated sums of gradient on individual samples 
    # updates latters based on formula of backward pass 

    # TODO memory is preallocated 
    # remember v is x2 (output of last node) 

    dl_dv = 2*(x2-t)
    print('shape dl_dv', dl_dv.shape)

    # dv_ds has dimensions ?
    dv_ds1 = dsigma(s1)
    print('shape dv_ds', dv_ds1.shape)
    dv_ds2 = dsigma(s2)
    dl_ds1 = torch.dot(dl_dv, dv_ds1)
    dl_ds2 = torch.dot(dl_dv, dv_ds2)

    dl_dw1 = torch.mv(dl_ds, x.t())
    dl_dw2 = torch.mv(dl_ds, x1.t())

    dl_db1 = dl_ds1 
    dl_db2 = dl_ds2

    return dl_dw1, dl_db1, dl_dw2, dl_db2 
    

def step(x, t,
        w1, b1, w2, b2, 
        dl_dw1, dl_db1, dl_dw2, dl_db2,
        step_size): 
    # forward pass 
    x0, s1, x1, s2, x2 = forward_pass(
            w1, b1, w2, b2, x)

    # backward pass
    dl_dw1, dl_db1, dl_dw2, dl_db2 =\
            backward_pass(
                    w1, b1, w2, b2,
                    t, 
                    x0, s1, x1, s2, x2, 
                    dl_dw1, dl_db1, dl_dw1, dl_db2
                    )
    # update in memory of initial params 

    # update params 
    w1 -= step_size * dl_dw1
    w2 -= step_size * dl_dw2
    b1 -= step_size * dl_db1
    b2 -= step_size * dl_db2
    
    return w1, b1, w2, b2


if __name__ == "__main__":
    # load data 
    x, t, x_test, t_test = prologue.load_data(
            one_hot_labels=True,
            normalize=True
            )
    x = x[0]
    t = t[0]

    # multiply target label vectors by 0.9 to make sure theyre in the range of tanh 
    t = t * 0.9
    t_test = t_test * 0.9

    n_layer_1 = 50
    n_layer_2 = 10

    # create four weight and bias tensors 
    # fill with random values samples from normal distribution N(0, 1e-6)
    w1 = torch.empty(n_layer_1, x.shape[0]).normal_(mean=0, std=1e-6)
    b1 = torch.empty(n_layer_1).normal_(mean=0, std=1e-6) 
    w2 = torch.empty(n_layer_2, n_layer_1).normal_(mean=0, std=1e-6)
    b2 = torch.empty(n_layer_2).normal_(mean=0, std=1e-6)

    # create tensors to sum up gradients 
    dl_dw1 = torch.zeros(w1.shape)
    dl_db1 = torch.zeros(b1.shape)
    dl_dw2 = torch.zeros(w2.shape)
    dl_db2 = torch.zeros(b2.shape)

    # perform 1k gradient steps with step size = 0.1 / N_train 
    step_size = 0.1 / t.shape[0]

    # first do step once 
    w1, b1, w2, b2 = step(x, t, w1, b1, w2, b2, 
                          dl_dw1, dl_db1, dl_dw1, dl_db2, step_size)


