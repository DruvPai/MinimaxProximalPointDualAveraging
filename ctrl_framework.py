import torch
import pytorch_lightning as pl
import mcr2

from operators.two_layer_networks import *
from optimizers.dummy_optimizer import *


optimizer_dict = {
    "mirror": DummyMinimaxOptimizer,
    "op_ex": DummyMinimaxOptimizer,
    "pdhg": DummyMinimaxOptimizer
}


class CTRL(pl.LightningModule):
    def __init__(self, d_x: int, d_z: int, k: int, eps_sq: float, latent_dim_f: int, latent_dim_g: int,
                 lr_f: float, lr_g: float, optimizer_name: str="mirror"):
        super(CTRL, self).__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.k = k

        self.lr_f = lr_f
        self.lr_g = lr_g
        self.optimizer_name = optimizer_name

        self.f = TwoLayerFCNet(d_x, d_z, latent_dim_f)
        self.g = TwoLayerFCNet(d_z, d_x, latent_dim_g)

        self.cr = mcr2.coding_rate.SupervisedVectorCodingRate(eps_sq)

    def Z(self, X):
        return self.f(X)

    def Z_hat(self, X):
        return self.f(self.g(self.f(X)))

    def forward(self, X):
        return self.Z(X)

    def training_step(self, batch, batch_idx):
        X, y = batch
        Z = self.Z(X)
        Z_hat = self.Z_hat(X)
        Pi = mcr2.functional.y_to_pi(y, self.k)
        return self.cr.DeltaR(Z, Pi) + self.cr.DeltaR(Z_hat, Pi) + sum(
            self.cr.DeltaR_distance(Z[y == j], Z_hat[y == j])
            for j in range(self.k)
        )

    def configure_optimizers(self):
        return optimizer_dict[self.optimizer_name](self.f.parameters(), self.g.parameters(), self.lr_f, self.lr_g)
