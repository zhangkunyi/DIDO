import torch


"""
MSE loss between prediction and target, no covariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""

l1loss = torch.nn.L1Loss()
def loss_mse(pred, targ):
    loss = (pred - targ).pow(2)
    return loss

def loss_e(pred, targ):
    loss = torch.exp((pred - targ).pow(2))
    return loss

"""
Log Likelihood loss, with covariance (only support diag cov)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_cov: Nx3 vector of log(sigma) on the diagonal entries
output:
  loss: Nx3 vector of likelihood loss on x,y,z

resulting pred_cov meaning:
pred_cov:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
"""


def loss_distribution_diag(pred, pred_cov, targ):
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_cov)) + pred_cov
    return loss

"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""

def get_loss(pred, pred_cov, targ, epoch):
    if epoch < 20:
        loss = loss_mse(pred, targ)
    else:
        loss = loss_distribution_diag(pred, pred_cov, targ)
    return loss
