import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import random
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.linear_model import LogisticRegression


np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MLPProbe(nn.Module):
    
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, x):
        h = F.relu(self.linear1(x))
        o = self.linear2(h)
        return torch.sigmoid(o)

class CCS(object):

    """Contrast Consistent Search (CCS):  a probe for analyzing a statement (x1) and the negation of the statement (x0),
    with classification task loss (optinal).
     
    Hyperparams:
    -----------------------------
    x0, x1 : np.ndarray
        Inputs: (x0) negation of that statement and the statement (x1)
        Example: x1 — Cats are mammals. Yes. 
                 x0 — Cats are mammals. No.
        
    !!!! Must be normalized after extraction with a formula x0 - x0.mean(0), x1-x1.mean(0)

    y_train : np.ndarray or None
        True labels for binary classification.
    nepochs : int, default=1500
        N epoch during each train().
    ntries : int, default=10
        Number of training runs with different initializations.
    lr : float, default=0.015
        learning rate.
    batch_size : int, default=-1
        Batch size, -1 means use all data (full batch).
    device : str or torch.device, default=None
        Device ('cpu' or 'cuda'),for training
    linear : bool, default=True
        Linear (True) or MLP (False) probe.
    weight_decay : float, default=0.01
       L2 regularization.
    var_normalize : bool, default=False
        If need normalize the data by standard deviation
    lambda_classification : float, default=0.0
        BCE weight. If 0 then we learn classic CCS
    
    """

    def __init__(self, x0, x1, y_train=None, nepochs=1500, ntries=10, lr=0.015, batch_size=-1,
                 device=None, linear=True, weight_decay=0.01, var_normalize=False, lambda_classification=0.0, predict_normalize=False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
            self.device = device
        
        self.var_normalize = var_normalize
        self.x0 = self.normalize(x0)
        self.x1 = self.normalize(x1)
        self.y_train = y_train
        self.d = self.x0.shape[-1]
        self.lambda_classification = lambda_classification

        # training params
        self.nepochs = nepochs
        self.ntries = ntries
        self.lr = lr
        self.device = device
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.linear = linear
        self.initialize_probe()
        self.best_probe = copy.deepcopy(self.probe)

    def initialize_probe(self):
        if self.linear:
            self.probe = nn.Sequential(nn.Linear(self.d, 1), nn.Sigmoid())
        else:
            self.probe = MLPProbe(self.d)
        self.probe.to(self.device).to(dtype=torch.float32)

    def normalize(self, x):
        x = x - x.mean(axis=0, keepdims=True)
        if self.var_normalize:
            x = x / (x.std(axis=0, keepdims=True) + 1e-6)
        return x

    def get_tensor_data(self):
        x0 = torch.tensor(self.x0, dtype=torch.float32, device=self.device)
        x1 = torch.tensor(self.x1, dtype=torch.float32, device=self.device)
        return x0, x1

    def get_loss(self, p0, p1, y_true=None):
        informative_loss = (torch.min(p0, p1) ** 2).mean()
        consistent_loss = ((p0 - (1 - p1)) ** 2).mean()

        if self.lambda_classification != 0 and y_true is not None:

            avg_pred = 0.5 * (p0 + (1 - p1))
            if not isinstance(y_true, torch.Tensor):
                y_true = torch.tensor(y_true, dtype=torch.float32, device=avg_pred.device)
            if y_true.ndim == 1:
                y_true = y_true.view(-1, 1)

            bce = nn.BCELoss()
            classification_loss = bce(avg_pred, y_true)

            return informative_loss + consistent_loss + self.lambda_classification*classification_loss

        return informative_loss + consistent_loss


    def predict(self, x0_test, x1_test, predict_normalize=False):
        
        if predict_normalize:
            x0_test = self.normalize(x0_test)
            x1_test = self.normalize(x1_test)

        x0 = torch.tensor(x0_test, dtype=torch.float32, device=self.device)
        x1 = torch.tensor(x1_test, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            p0, p1 = self.best_probe(x0), self.best_probe(x1)
        avg_confidence = 0.5 * (p0 + (1 - p1))
        predictions = (avg_confidence.cpu().numpy() > 0.5).astype(int)[:, 0]

        return predictions, avg_confidence

    def get_acc(self, x0_test, x1_test, y_test):
  
        predictions, _ = self.predict(x0_test, x1_test)
        acc = (predictions == y_test).mean()

        return max(acc, 1 - acc)
    

    def get_silhouette(self, x0_test, x1_test):

        predictions, _ = self.predict(x0_test, x1_test)
        s_score = 0 if len(np.unique(predictions)) == 1 else silhouette_score(x1_test - x0_test, predictions, metric='cosine')
        return s_score

    def get_contrastive_probas(self, xA0_test, xA1_test, x_notA0_test, x_notA1_test):

        xA0_test = torch.tensor(self.normalize(xA0_test), dtype=torch.float32, device=self.device)
        xA1_test = torch.tensor(self.normalize(xA1_test), dtype=torch.float32, device=self.device)

        x_notA0_test = torch.tensor(self.normalize(x_notA0_test), dtype=torch.float32, device=self.device)
        x_notA1_test = torch.tensor(self.normalize(x_notA1_test), dtype=torch.float32, device=self.device)
        
        with torch.no_grad():

          pA0, pA1 = self.best_probe(xA0_test).detach().cpu().numpy(), self.best_probe(xA1_test).detach().cpu().numpy()
          p_notA0, p_notA1 = self.best_probe(x_notA0_test).detach().cpu().numpy(), self.best_probe(x_notA1_test).detach().cpu().numpy()

        return pA0, pA1, p_notA0, p_notA1

    def get_agreement(self, xA0_test, xA1_test, x_notA0_test, x_notA1_test):

        pA0, pA1, p_notA0, p_notA1 = self.get_contrastive_probas(xA0_test, xA1_test, x_notA0_test, x_notA1_test)

        agreement_score = 0.5*((pA1 - p_notA0)**2 + (pA0 - p_notA1)**2) * np.sign(pA1 - p_notA1) * np.sign(p_notA0 - pA0)
        return agreement_score

    def get_contradiction_idx(self, xA0_test, xA1_test, x_notA0_test, x_notA1_test):

        pA0, pA1, p_notA0, p_notA1 = self.get_contrastive_probas(xA0_test, xA1_test, x_notA0_test, x_notA1_test)
        ci = pA1*p_notA1 + pA0*p_notA0

        return ci


    def train(self):

        x0, x1 = self.get_tensor_data()
        batch_size = len(x0) if self.batch_size == -1 else self.batch_size
        nbatches = len(x0) // batch_size

        if self.lambda_classification != 0 and self.y_train is not None:
            y_tensor = torch.tensor(self.y_train, dtype=torch.float32, device=self.device)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.view(-1, 1)

        self.probe = self.probe.to(self.device).to(torch.float32)
        optimizer = optim.AdamW(self.probe.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.nepochs):
            permutation = torch.randperm(len(x0), device=self.device)
            x0, x1 = x0[permutation], x1[permutation]

            if self.lambda_classification != 0 and self.y_train is not None:
                y_tensor = y_tensor[permutation]

            for j in range(nbatches):
                x0_batch = x0[j * batch_size:(j + 1) * batch_size].to(self.device).to(torch.float32)
                x1_batch = x1[j * batch_size:(j + 1) * batch_size].to(self.device).to(torch.float32)
                y_batch = y_tensor[j * batch_size:(j + 1) * batch_size] if self.lambda_classification !=0 and self.y_train is not None else None
                if y_batch is not None:
                    y_batch = y_batch.to(self.device).to(torch.float32)

                p0, p1 = self.probe(x0_batch), self.probe(x1_batch)
                loss = self.get_loss(p0, p1, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return loss.detach().cpu().item()
    
    def get_weights(self):
        """
        Returns the learned weights and bias of the linear probe.
        Only works if linear=True.
        Returns:
            weight: numpy array of shape (d,)
            bias: float
        """
        if not self.linear:
            raise ValueError("Weights can only be extracted from linear probes.")

        linear_layer = self.best_probe[0]  # nn.Linear layer
        weight = linear_layer.weight.detach().cpu().numpy().squeeze()  # shape: (1, d) -> (d,)
        bias = linear_layer.bias.detach().cpu().numpy().item()

        return weight, bias


    def repeated_train(self):
      
        best_loss = np.inf
        for train_num in range(self.ntries):
            self.initialize_probe()
            loss = self.train()
            if loss < best_loss:
                self.best_probe = copy.deepcopy(self.probe)
                best_loss = loss
        return best_loss

def train_lr_on_hidden_states(X_pos, X_neg, y_vec, train_idx, test_idx, random_state=71):
    """
    Train baseline (logistic regression) on hidden states

    Parameters:
        X_pos (np.ndarray): Positive statements shape (N, n_layers, hidden_dim).
        X_neg (np.ndarray): Negatize statements, shape (N, n_layers, hidden_dim).
        
        !!!! Must be NOT normalized after extraction

        y_vec (np.ndarray): y labels
        train_idx (np.ndarray): train indexes
        test_idx (np.ndarray): test indexes

        random_state (int): Random seed.

    Returns:
        results (dict): Dict {Layer number: {'Test Accuracy': ,
                               'Test Silhouette score' :}}.
    """
    n_samples, n_layers, hidden_dim = X_pos.shape
    results = {}

    X_pos = X_pos - X_pos.mean(0)
    X_neg = X_neg - X_neg.mean(0)


    for layer_idx in range(n_layers):

        # X_pos, X_neg train test split
        X_pos_train_layer = X_pos[train_idx, layer_idx, :]  # (train_samples, hidden_dim)
        X_pos_test_layer = X_pos[test_idx, layer_idx, :]

        X_neg_train_layer = X_neg[train_idx, layer_idx, :]
        X_neg_test_layer = X_neg[test_idx, layer_idx, :]

        # y vector
        y_train = y_vec[train_idx]
        y_test = y_vec[test_idx]

        # Preparing to LR training
        X_train_layer = (X_pos_train_layer - X_neg_train_layer)
        X_test_layer  = (X_pos_test_layer - X_neg_test_layer)

        # Train Lr
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train_layer, y_train)

        # Predictions ans scoring (silhouette score only if lr found more that 1 class)
        y_pred = clf.predict(X_test_layer)
        acc = accuracy_score(y_test, y_pred)

        if len(np.unique(y_pred)) == 1:
          s_score = 0
        else:
          s_score = silhouette_score(X_test_layer, y_pred, metric='cosine')


        # Save results
        results[layer_idx] = {'Accuracy' : acc,
                              'Silhouette' : s_score}

    return results

def train_ccs_on_hidden_states(X_pos, X_neg, y_vec, train_idx, 
                               test_idx, lambda_classification=0.0, normalizing='mean', device=None):
    """
    Train CCS for each layer and get results 

    Parameters:
        X_pos (np.ndarray): Positive (yes) samples, shape (N, n_layers, hidden_dim). Must be raw (not normalized after extraction)
        X_neg (np.ndarray): Negative (no) samples, shape (N, n_layers, hidden_dim). Must be raw (not normalized after extraction)
        y_vec (np.ndarray): y labels 
        train_idx (np.ndarray): train indexes
        test_idx (np.ndarray): test indexes

        random_state (int): Random seed.
        lambda_classification (float): BCE weight
        
        normalize (str): {'mean', 'median', 'l2', 'raw', None}
               — mean 
               X_train := X_train - X_train.mean(axis=0) 
               X_test  := X_test  - X_train.mean(axis=0)
               - median
               X_train := X_train - X_train.median(axis=0)  
               X_test  := X_test  - X_train.median(axis=0)
               - l2 
                X := X / ||X||₂
                - raw or None — without normalization

    Returns:
        results (dict): dict {layer number: {'accuracy': ccs_acc,
                              'silhouette' : s_score,
                              'agreement' : ccs_agreement,
                              'contradiction idx' : ccs_ci,
                              'IM dist': ccs_ideal_dist}
 }.
    """
    n_samples, n_layers, hidden_dim = X_pos.shape
    results = {}


    for layer_idx in tqdm(range(n_layers), desc="Training CCS on hidden states"):

        # X positive (yes)
        X_pos_train_layer = X_pos[train_idx, layer_idx, :].astype(np.float32) # (train_samples, hidden_dim)
        X_pos_test_layer = X_pos[test_idx, layer_idx, :].astype(np.float32)

        # X negative (no)
        X_neg_train_layer = X_neg[train_idx, layer_idx, :].astype(np.float32)
        X_neg_test_layer = X_neg[test_idx, layer_idx, :].astype(np.float32)


        if normalizing == 'mean':
            # print('Mean normalize used')
            X_pos_train_mean = X_pos_train_layer.mean(0)
            X_neg_train_mean = X_neg_train_layer.mean(0)

            X_pos_train_layer = X_pos_train_layer - X_pos_train_mean
            X_pos_test_layer =X_pos_test_layer - X_pos_train_mean

            X_neg_train_layer = X_neg_train_layer - X_neg_train_mean
            X_neg_test_layer =X_neg_test_layer - X_neg_train_mean

        if normalizing == 'median':
            # print('Median normalize used')
            X_pos_train_median = np.median(X_pos_train_layer, 0)
            X_neg_train_median = np.median(X_neg_train_layer, 0)

            X_pos_train_layer = X_pos_train_layer - X_pos_train_median
            X_pos_test_layer =X_pos_test_layer - X_pos_train_median

            X_neg_train_layer = X_neg_train_layer - X_neg_train_median
            X_neg_test_layer =X_neg_test_layer - X_neg_train_median

        if normalizing == 'l2':
            # print('L2-only normalize used')
            X_pos_train_layer = normalize(X_pos_train_layer, norm='l2', axis=1)
            X_neg_train_layer = normalize(X_neg_train_layer, norm='l2', axis=1)

            X_pos_test_layer = normalize(X_pos_test_layer, norm='l2', axis=1)
            X_neg_test_layer = normalize(X_neg_test_layer, norm='l2', axis=1)

        if normalizing is None or normalize == 'raw':
            pass

        # y vector
        y_train = y_vec[train_idx].astype(np.float32)
        y_test = y_vec[test_idx].astype(np.float32)

        ccs = CCS(X_neg_train_layer, X_pos_train_layer, y_train.values, var_normalize=False, lambda_classification=lambda_classification, device=device)
        ccs.repeated_train()

        # Оценка
        predictions, conf = ccs.predict(X_neg_test_layer, X_pos_test_layer)
        if len(np.unique(predictions)) == 1:
          s_score = 0
        else:
          s_score = ccs.get_silhouette(X_neg_test_layer, X_pos_test_layer)
        ccs_acc = ccs.get_acc(X_neg_test_layer, X_pos_test_layer, y_test.values)


        # print(f"Layer {layer_idx+1}/{n_layers}, CCS accuracy: {ccs_acc}")

        # Probas
        A_idx = test_idx[test_idx >= len(X_pos)/2]
        notA_idx = (A_idx - n_samples/2).astype(int)

        A0_test = X_neg[A_idx, layer_idx, :]
        A1_test = X_pos[A_idx, layer_idx, :]

        notA0_test = X_neg[notA_idx, layer_idx, :]
        notA1_test = X_pos[notA_idx, layer_idx, :]

        ccs_agreement = ccs.get_agreement(A0_test, A1_test, notA0_test, notA1_test)
        ccs_ci = ccs.get_contradiction_idx(A0_test, A1_test, notA0_test, notA1_test)

        # Save result
        results[layer_idx] = {'accuracy': ccs_acc,
                              'silhouette' : s_score,
                              'agreement' : ccs_agreement,
                              'contradiction idx' : ccs_ci,
                             'weights' : ccs.get_weights()[0],
                              'bias' : ccs.get_weights()[1]}
    return results

def train_half_ccs_on_hidden_states(X_pos, X_neg, y_vec, random_state=71, lambda_classification=0.0, weight_decay=0.01, normalize=True, device=None):
    """
    Train CCS for each layer and get results 

    Parameters:
        X_pos (np.ndarray): Positive (yes) samples, shape (N, n_layers, hidden_dim). Must be raw (not normalized after extraction)
        X_neg (np.ndarray): Negative (no) samples, shape (N, n_layers, hidden_dim). Must be raw (not normalized after extraction)
        y_vec (np.ndarray): y labels 
        train_idx (np.ndarray): train indexes
        test_idx (np.ndarray): test indexes

        random_state (int): Random seed.
        lambda_classification (float): BCE weight
        
        normalize (bool): if True then training data normalized with formula X - X.mean(0) else raw data is used 
        (only if you have normalization before)

    Returns:
        results (dict): dict {layer number: {'accuracy': ccs_acc,
                              'silhouette' : s_score,
                              'agreement' : ccs_agreement,
                              'contradiction idx' : ccs_ci,
                              'IM dist': ccs_ideal_dist}
 }.
    """
    n_samples, n_layers, hidden_dim = X_pos.shape
    results = {}

    if normalize:
        X_pos = X_pos - X_pos.mean(0)
        X_neg = X_neg - X_neg.mean(0)

    all_idx = np.arange(len(X_pos)//2) # All idxs 

    train_idx, test_idx = train_test_split(all_idx, test_size=0.15, random_state=71, shuffle=True)

    first_pos_cluster = X_pos[:len(X_pos)//2, :]
    second_pos_cluster = X_pos[len(X_pos)//2:, :]

    first_neg_cluster = X_neg[:len(X_neg)//2, :]
    second_neg_cluster = X_neg[len(X_neg)//2:, :]


    for layer_idx in range(n_layers):

        # first cluster pos
        X_pos_first_cluster_train_layer = first_pos_cluster[train_idx, layer_idx, :].astype(np.float32) # (train_samples, hidden_dim)
        X_pos_first_cluster_test_layer = first_pos_cluster[test_idx, layer_idx, :].astype(np.float32)

        # second cluster pos
        X_pos_second_cluster_train_layer = second_pos_cluster[train_idx, layer_idx, :].astype(np.float32) # (train_samples, hidden_dim)
        X_pos_second_cluster_test_layer = second_pos_cluster[test_idx, layer_idx, :].astype(np.float32)

        # first cluster neg
        X_neg_first_cluster_train_layer = first_neg_cluster[train_idx, layer_idx, :].astype(np.float32) # (train_samples, hidden_dim)
        X_neg_first_cluster_test_layer = first_neg_cluster[test_idx, layer_idx, :].astype(np.float32)

        # second cluster neg
        X_neg_second_cluster_train_layer = second_neg_cluster[train_idx, layer_idx, :].astype(np.float32) # (train_samples, hidden_dim)
        X_neg_second_cluster_test_layer = second_neg_cluster[test_idx, layer_idx, :].astype(np.float32)


        # y first
        y_first = y_vec[:len(X_neg)//2]
        y_second = y_vec[len(X_neg)//2:]
       
        # y first
        y_train_first = y_first[train_idx].astype(np.float32)
        y_test_first = y_first[test_idx].astype(np.float32)
        
        # y second
        y_train_second = y_second[train_idx].astype(np.float32)
        y_test_second = y_second[test_idx].astype(np.float32)



        ccs_first = CCS(X_neg_first_cluster_train_layer, X_pos_first_cluster_train_layer, y_train_first, 
                        var_normalize=False, weight_decay=weight_decay, lambda_classification=lambda_classification, device=device)
        ccs_first.repeated_train()

        ccs_second = CCS(X_neg_second_cluster_train_layer, X_pos_second_cluster_train_layer, y_train_second, 
                        var_normalize=False, weight_decay=weight_decay, lambda_classification=lambda_classification, device=device)
        ccs_second.repeated_train()


        # Оценка
        predictions_first, conf_first = ccs_first.predict(X_neg_first_cluster_test_layer, X_pos_first_cluster_test_layer)
        predictions_second, conf_second = ccs_first.predict(X_neg_first_cluster_test_layer, X_pos_first_cluster_test_layer)

        if len(np.unique(predictions_first)) == 1:
            s_score_f = 0
        else:
            s_score_f = ccs_first.get_silhouette(X_neg_first_cluster_test_layer, X_pos_first_cluster_test_layer)

        if len(np.unique(predictions_second)) == 1:
            s_score_s = 0
        
        else:
            s_score_s = ccs_second.get_silhouette(X_neg_second_cluster_test_layer, X_pos_second_cluster_test_layer)

        weights_f = ccs_first.get_weights()
        weights_s = ccs_second.get_weights()

        print(f'Layer : {layer_idx}/{n_layers} finished')



        # Save result
        results[layer_idx] = {'s_score_f': s_score_f,
                              's_score_s': s_score_s,
                              'weights_f' : weights_f, 
                              'weights_s' : weights_s
                              }

    return results

def train_lr_on_hidden_states(X_pos, X_neg, y_vec, train_idx, test_idx, random_state=71):
    """
    Train baseline (logistic regression) on hidden states

    Parameters:
        X_pos (np.ndarray): Positive statements shape (N, n_layers, hidden_dim).
        X_neg (np.ndarray): Negatize statements, shape (N, n_layers, hidden_dim).
        train_idx (np.ndarray): train indexes
        test_idx (np.ndarray): test indexes

        random_state (int): Random seed.

    Returns:
        results (dict): Dict {Layer number: {'Test Accuracy': ,
                               'Test Silhouette score' :}}.
    """
    n_samples, n_layers, hidden_dim = X_pos.shape
    results = {}

    X_pos = X_pos - X_pos.mean(0)
    X_neg = X_neg - X_neg.mean(0)


    for layer_idx in range(n_layers):

        # X_pos, X_neg train test split
        X_pos_train_layer = X_pos[train_idx, layer_idx, :]  # (train_samples, hidden_dim)
        X_pos_test_layer = X_pos[test_idx, layer_idx, :]

        X_neg_train_layer = X_neg[train_idx, layer_idx, :]
        X_neg_test_layer = X_neg[test_idx, layer_idx, :]

        # y vector
        y_train = y_vec[train_idx]
        y_test = y_vec[test_idx]

        # Preparing to LR training
        X_train_layer = (X_pos_train_layer - X_neg_train_layer)
        X_test_layer  = (X_pos_test_layer - X_neg_test_layer)

        # Train Lr
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
        clf.fit(X_train_layer, y_train)

        # Predictions ans scoring (silhouette score only if lr found more that 1 class)
        y_pred = clf.predict(X_test_layer)
        acc = accuracy_score(y_test, y_pred)

        if len(np.unique(y_pred)) == 1:
          s_score = 0
        else:
          s_score = silhouette_score(X_test_layer, y_pred, metric='cosine')


        # Save results
        results[layer_idx] = {'Accuracy' : acc,
                              'Silhouette' : s_score}

    return results