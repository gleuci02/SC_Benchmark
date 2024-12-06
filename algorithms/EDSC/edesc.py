from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
import warnings
from .AutoEncoder import AE
from .InitializeD import Initialization_D
from .Constraint import D_constraint1, D_constraint2
import time
warnings.filterwarnings("ignore")
   
class EDESC(nn.Module):

    def __init__(self,
                 n_enc_1,
                 n_enc_2,
                 n_enc_3,
                 n_dec_1,
                 n_dec_2,
                 n_dec_3,
                 n_input,
                 n_z,
                 n_clusters,
                 num_sample,
                 pretrain_path='data/reuters.pkl'):
        super(EDESC, self).__init__()
        self.pretrain_path = pretrain_path
        self.n_clusters = n_clusters

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)	

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(n_z, n_clusters))

        
    def pretrain(self, path=''):
        if path == '':
            pretrain_ae(self.ae)
        # Load pre-trained weights
        self.ae.load_state_dict(torch.load(self.pretrain_path, map_location='cpu'))
        print('Load pre-trained model from', path)

    def forward(self, x):
        
        x_bar, z = self.ae(x)
        d = args.d
        s = None
        eta = args.eta
      
        # Calculate subspace affinity
        for i in range(self.n_clusters):	
			
            si = torch.sum(torch.pow(torch.mm(z,self.D[:,i*d:(i+1)*d]),2),1,keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s,si),1)   
        s = (s+eta*d) / ((eta+1)*d)
        s = (s.t() / torch.sum(s, 1)).t()
        return x_bar, s, z

    def total_loss(self, x, x_bar, z, pred, target, dim, n_clusters, beta):

	# Reconstruction loss
        reconstr_loss = F.mse_loss(x_bar, x)     
        
        # Subspace clustering loss
        kl_loss = F.kl_div(pred.log(), target)
        
        # Constraints
        d_cons1 = D_constraint1()
        d_cons2 = D_constraint2()
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, dim, n_clusters)
  
        # Total_loss
        total_loss = reconstr_loss + beta * kl_loss + loss_d1 + loss_d2

        return total_loss
    
    def fit(self, X):
        """
        Train the EDESC model.
        Args:
            X: Input data as a NumPy array.
        """
        args = args
        device = torch.device("cpu")
        X = torch.tensor(X).float().to(device)
        optimizer = Adam(self.model.parameters(), lr=self.config["lr"])

        # Pretrain autoencoder
        self.model.pretrain(self.config.get("MNIST", "data/mnist/MNIST/raw/train-images-idx3-ubyte"))

        # Initialize clustering with KMeans
        with torch.no_grad():
            x_bar, hidden = self.model.ae(X)
        kmeans = KMeans(n_clusters=self.config["n_clusters"], n_init=10)
        y_pred = kmeans.fit_predict(hidden.cpu().numpy())

        # Initialize D
        D = Initialization_D(hidden, y_pred, self.config["n_clusters"], self.config["d"])
        self.model.D.data = torch.tensor(D).float().to(self.device)

        # Training loop
        for epoch in range(self.config["epochs"]):
            x_bar, s, z = self.model(X)
            s_tilde = self._refined_subspace_affinity(s)

            # Loss calculation
            loss = self.model.total_loss(
                X, x_bar, z, pred=s, target=s_tilde, dim=self.config["d"], n_clusters=self.config["n_clusters"], beta=self.config["beta"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
    def predict(self, X):
        """
        Predict cluster labels using the trained EDESC model.
        Args:
            X: Input data as a NumPy array.
        Returns:
            Predicted cluster labels as a NumPy array.
        """
        device = torch.device("cpu")
        X = torch.tensor(X).float().to(device)
        with torch.no_grad():
            _, s, _ = self.model(X)
        return torch.argmax(s, dim=1).cpu().numpy()

def pretrain_ae(model):

    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    for epoch in range(50):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), args.pretrain_path)
    print("Model saved to {}.".format(args.pretrain_path))





 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='EDESC training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--d', default=5, type=int)
    parser.add_argument('--n_z', default=20, type=int)
    parser.add_argument('--eta', default=5, type=int)
    parser.add_argument('--dataset', type=str, default='reuters')
    parser.add_argument('--pretrain_path', type=str, default='data/reuters')
    parser.add_argument('--beta', default=0.1, type=float, help='coefficient of subspace affinity loss')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.dataset = 'reuters'
    if args.dataset == 'reuters':
        args.pretrain_path = 'data/reuters.pkl'
        args.n_clusters = 4
        args.n_input = 2000
        args.num_sample = 10000
        dataset = LoadDataset(args.dataset)   
    print(args)
    bestacc = 0 
    bestnmi = 0
    for i in range(10):
        acc, nmi = train_EDESC()
        if acc > bestacc:
            bestacc = acc
        if nmi > bestnmi:
            bestnmi = nmi
    print('Best ACC {:.4f}'.format(bestacc), ' Best NMI {:4f}'.format(bestnmi))