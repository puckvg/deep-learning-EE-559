import torch 
from torch import Tensor 
import practical_prologue as prologue 


def nearest_classification(train_input, train_target, x): 
    """
    Get a training set and a test sample and return the label
    of the training point closest to the test

    Parameters
    ----------
    train_input : 2d float tensor 
                  dimensions n x d 
                  containing training vectors 
    train_target : 1d long tensor 
                   dimension n 
                   containing training labels 
    x : 1d float tensor 
        dimension d 
        contains test vector 

    Returns 
    -------
    y : long tensor 
        the class of the train sample closest to x for the L2 norm
    """
    L2 = torch.sum(torch.pow((train_input - x), 2), 1)
    sorted_norm, indices = torch.sort(L2)
    y_pred = train_target[indices[0]]

    return y_pred 


def compute_nb_errors(train_input, train_target, test_input, test_target,
                      mean=None, proj=None):
    """
    Take vectors train_input and test_input, apply operator proj (if it is 
    not None) to both and return the number of classification errors using
    the 1-nearest-neighbour rule on the resulting data 

    Parameters 
    ----------
    train_input : 2d float tensor 
                  dimension n x d containing train vectors 
    train_target : 1d long tensor 
                   dimension n containing train labels 
    test_input : 2d float tensor 
                 dimension m x d containing test vectors 
    test_target : 1d long tensor 
                  dimension m containing test labels 
    mean : None or 1d float tensor 
           1d float tensor dimension d 
    proj : 2d float tensor
           dim cxd
           used as a basis to reconstruct the input data 

    Returns
    -------
    err : number of classification errors
    """
    if mean is not None: 
        train_input = train_input - mean 
        test_input = test_input - mean 

    if proj is not None:
        train_input = torch.mm(train_input, proj.t())
        test_input = torch.mm(test_input, proj.t()) 

    y_pred = torch.empty(test_input.shape[0])
    for i, x in enumerate(test_input): 
        y_pred[i] = nearest_classification(train_input, train_target, x)

    # count number of incorrect classifications by comparing to test_target 
    incorrect = (y_pred != test_target).sum()

    return incorrect


def PCA(x): 
    """
    Take a 2D float tensor and return the mean and the PCA basis ranked 
    in decreasing order of eigenvalues 

    Parameters 
    ----------
    x : 2d float tensor 
        dimension n x d

    Returns 
    -------
    mean : 1d vector dimension d
    basis : 2d tensor dxd 
            mean vector and PCA basis ranked in decreasing order
            of eigenvalues
    """
    mean = torch.mean(train_input, 0)

    cov = (1 / (x.size()[1]) -1 ) * torch.mm((x-mean).t(), (x-mean)) 
    # get PCA basis from eigendecomposition 
    eigvals, eigvecs = torch.eig(cov, eigenvectors=True)
    # keep real data -is this ok ? 
    eigvals = eigvals[:,0]
    # sorted in decreasing order of eigenvalues 
    sorted_eigvals, indices = torch.sort(torch.abs(eigvals), descending=True)
    basis = eigvecs[indices]

    return mean, basis


if __name__ == "__main__":
    # FIRST MNIST
    train_input, train_target, test_input, test_target = prologue.load_data()

    # project data on random 100d subspace 
    mean = torch.mean(train_input, 0)
    random_basis = torch.empty((100, train_input.shape[1])).normal_()
    n_incorrect_random = compute_nb_errors(train_input, train_target, test_input, 
                                    test_target, mean=mean, proj=random_basis)
    print('n incorrect for random projection', n_incorrect_random)

    for n in [3, 10, 50, 100]:
        mean, basis = PCA(train_input) 
        basis = basis[:n]
        n_incorrect_pca_n = compute_nb_errors(train_input, train_target, test_input, 
                                            test_target, mean=mean, proj=basis)
        print('n incorrect for pca with {} dim'.format(n), n_incorrect_pca_n)
    

    # this is suspicious. probably did something wrong. shouldn't get so many wrong  
    # THEN CIFAR 
    train_input, train_target, test_input, test_target = prologue.load_data(cifar=True)

    # project data on random 100d subspace 
    mean = torch.mean(train_input, 0)
    random_basis = torch.empty((100, train_input.shape[1])).normal_()
    n_incorrect_random = compute_nb_errors(train_input, train_target, test_input, 
                                    test_target, mean=mean, proj=random_basis)
    print('n incorrect for random projection', n_incorrect_random)

    for n in [3, 10, 50, 100]:
        mean, basis = PCA(train_input) 
        basis = basis[:n]
        n_incorrect_pca_n = compute_nb_errors(train_input, train_target, test_input, 
                                            test_target, mean=mean, proj=basis)
        print('n incorrect for pca with {} dim'.format(n), n_incorrect_pca_n)
