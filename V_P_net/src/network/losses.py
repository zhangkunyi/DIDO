
import torch

from torch.nn import functional as F



"""
MSE loss between prediction and target, no covariance

input: 
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
output:
  loss: Nx3 vector of MSE loss on x,y,z
"""

def loss_mse_R3(pred, targ):
    loss = (pred - targ).pow(2)
    return loss



def loss_derivate(pred,old_pred):
    try:
        loss = torch.abs(pred - old_pred)
    except:
        print('AA')
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


def loss_distribution_diag_R3(pred, pred_cov, targ):
    loss = ((pred - targ).pow(2)) / (2 * torch.exp(2 * pred_cov) )+ pred_cov

    return loss


"""
Log Likelihood loss, with covariance (support full cov)
(NOTE: output is Nx1)

input:
  pred: Nx3 vector of network displacement output
  targ: Nx3 vector of gt displacement
  pred_cov: Nxk covariance parametrization
output:
  loss: Nx1 vector of likelihood loss

resulting pred_cov meaning:
DiagonalParam:
pred_cov:(Nx3) u = [log(sigma_x) log(sigma_y) log(sigma_z)]
PearsonParam:
pred_cov (Nx6): u = [log(sigma_x) log(sigma_y) log(sigma_z)
                     rho_xy, rho_xz, rho_yz] (Pearson correlation coeff)
FunStuff
"""

"""
logrithm map of a normalised quaternion
input:
  q: a normalised quaternion in pyquaternion
output:
  loss: batch x 3 tensor in torch
"""

def q_log_torch(q):
    if len(q.shape) == 2:
        log = torch.zeros(q.shape[0],3).to(q.device)
        qv = q[:,1:]
        qvNorm = torch.norm(qv,p=2,dim=1,keepdim=True)
        w = q[:,0].unsqueeze(1)

        phi = torch.atan2(qvNorm,w)

        u0 = torch.zeros(phi.shape[0],3).to(q.device) # 对应于     if phi == 0: u = np.array([0.0, 0.0, 0.0])
        log = torch.where(phi==0,phi * u0,log)        # phi==0的行，把log相同的行换成phi * u0

        u1 = qv/w*(1-qvNorm*qvNorm/(3*w*w))           # 对应于     if phi == 0: u = qv/q.w*(1-qvNorm*qvNorm/(3*q.w*q.w))
        log = torch.where(phi < 1e-6, phi * u1, log)  # phi< 1e-6的行，把log相同的行换成phi * u1

        u2 = qv / qvNorm                              # 对应于     if phi == 0: u = qv / qvNorm
        log = torch.where(phi >= 1e-6, phi * u2, log) # phi>= 1e-6的行，把log相同的行换成phi * u2
    elif len(q.shape) == 3:
        log = torch.zeros(q.shape[0],q.shape[1], 3).to(q.device)
        qv = q[:,:, 1:]
        qvNorm = torch.norm(qv, p=2, dim=2, keepdim=True)
        if q.shape[1] != 1:
            qvNorm = torch.where(qvNorm == 0 ,torch.full_like(qvNorm,1),qvNorm )
        w = q[:,:, 0].unsqueeze(2)

        phi = torch.atan2(qvNorm,w)

        u0 = torch.zeros(phi.shape[0],phi.shape[1],3).to(q.device) # 对应于     if phi == 0: u = np.array([0.0, 0.0, 0.0])
        log = torch.where(phi==0,phi * u0,log)        # phi==0的行，把log相同的行换成phi * u0

        u1 = qv/w*(1-qvNorm*qvNorm/(3*w*w))           # 对应于     if phi == 0: u = qv/q.w*(1-qvNorm*qvNorm/(3*q.w*q.w))
        log = torch.where(phi < 1e-6, phi * u1, log)  # phi< 1e-6的行，把log相同的行换成phi * u1

        u2 = qv / qvNorm                              # 对应于     if phi == 0: u = qv / qvNorm
        log = torch.where(phi >= 1e-6, phi * u2, log) # phi>= 1e-6的行，把log相同的行换成phi * u

    return log


"""
Quaternion tensor multiplication
input: 
  q: Nx4 Multiply quaternion(s) q
  r: Nx4 Multiply quaternion(s) r
output:
  Returns q*r as a tensor of shape (*, 4).
"""
def qmul(q, r):
    """
    fork form https://github.com/facebookresearch/QuaterNet/blob/main/common/quaternion.py#L36
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    if len(q.shape) == 2:
        # Compute outer product
        terms = torch.bmm(r.contiguous().view(-1, 4, 1), q.contiguous().view(-1, 1, 4))

        w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
        x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
        y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
        z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
        return torch.stack((w, x, y, z), dim=1).view(original_shape).to(q.device)
    elif len(q.shape) == 3:
        terms = torch.matmul(r.contiguous().view(-1,q.shape[1], 4, 1), q.contiguous().view(-1,q.shape[1],  1, 4))

        w = terms[:,:, 0, 0] - terms[:,:, 1, 1] - terms[:,:, 2, 2] - terms[:,:, 3, 3]
        x = terms[:,:, 0, 1] + terms[:,:, 1, 0] - terms[:,:, 2, 3] + terms[:,:, 3, 2]
        y = terms[:,:, 0, 2] + terms[:,:, 1, 3] + terms[:,:, 2, 0] - terms[:,:, 3, 1]
        z = terms[:,:, 0, 3] - terms[:,:, 1, 2] + terms[:,:, 2, 1] + terms[:,:, 3, 0]
        return torch.stack((w, x, y, z), dim=2).view(original_shape).to(q.device)


"""
MSE loss between prediction and target, no covariance

input: 
  pred: Nx4 vector of network relative rotation output
  targ: Nx4 vector of gt relative rotation
output:
  loss: Nx3 vector of MSE loss on so(3)
"""
def loss_mse_so3_q(pred_q, targ_q):
    pred_q_normalized = F.normalize(pred_q,dim=len(pred_q.shape)-1)
    targ_q_normalized = F.normalize(targ_q,dim=len(pred_q.shape)-1)
    rela_q_normalized = F.normalize(qmul(pred_q_normalized * torch.tensor([1,-1,-1,-1]).to(pred_q.device),targ_q_normalized.to(pred_q.device)),dim=len(pred_q.shape)-1)
    e = q_log_torch(rela_q_normalized)
    loss = e.pow(2)
    return loss
"""
Quaternion Log Likelihood loss, with covariance (support full cov)
"""
def loss_distribution_diag_so3_q(pred_q, pred_q_cov, targ_q):
    pred_q_normalized = F.normalize(pred_q,dim=len(pred_q.shape)-1)
    targ_q_normalized = F.normalize(targ_q,dim=len(pred_q.shape)-1)
    rela_q_normalized = F.normalize(
        qmul(pred_q_normalized * torch.tensor([1, -1, -1, -1]).to(pred_q.device), targ_q_normalized),dim=len(pred_q.shape)-1)
    so3 = q_log_torch(rela_q_normalized)
    so3_square = so3.pow(2)
    loss = so3_square / (2 * torch.exp(2 * pred_q_cov)) + pred_q_cov
    return loss


def KL_Loss(mu,log_var):
    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    return kld_loss

"""
Select loss function based on epochs
all variables on gpu
output:
  loss: Nx3
"""
loss_fuc = torch.nn.SmoothL1Loss(reduction='none')

def get_loss(pred, pred_cov, targ, epoch,arch,targ_2=None):

    if arch in ["world_p_v_lstm_axis"]:
        if epoch > 20:
            loss_p = loss_distribution_diag_R3(pred[:,0], pred_cov[:,0], targ[:,0]) #/ targ.shape[0]
            loss_v = loss_distribution_diag_R3(pred[:,1], pred_cov[:,1], targ[:,1])
        else:
            loss_p = loss_mse_R3(pred[:,0], targ[:,0]) #/ targ.shape[0]
            loss_v = loss_mse_R3(pred[:,1], targ[:,1])          

        loss = loss_p  + loss_v
        print("train_double loss_p: ", torch.mean(loss_p, dim=0))
        print("train_double loss_v: ", torch.mean(loss_v, dim=0))
    else:
        raise ValueError("Invalid architecture to losses.py:", arch)

    return loss
