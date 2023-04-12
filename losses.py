import torch
from torch.nn import functional as F

import commons


def constractive_loss(hidden1: torch.Tensor,
                      hidden2: torch.Tensor,
                      hidden_norm: bool = True,
                      temperature: float = 1.0,
                      large_num: float = 1e9):
  """
  hidden1/hidden2: (T, B)
  """
  T, B = hidden1.shape

  if hidden_norm:
      hidden1 = torch.nn.functional.normalize(hidden1, p=2, dim=-1)
      hidden2 = torch.nn.functional.normalize(hidden2, p=2, dim=-1)

  hidden1_large = hidden1
  hidden2_large = hidden2
  labels = torch.arange(0, T).to(device=hidden1.device)

  logits_ab = torch.matmul(hidden1, hidden2_large.transpose(0, 1)) / temperature
  logits_ba = torch.matmul(hidden2, hidden1_large.transpose(0, 1)) / temperature

  loss_a = torch.nn.functional.cross_entropy(logits_ab, labels) # shape (T, T)
  loss_b = torch.nn.functional.cross_entropy(logits_ba, labels)
  loss = loss_a + loss_b
  return loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
