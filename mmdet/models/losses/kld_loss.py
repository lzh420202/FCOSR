import torch
from torch import nn
from ..builder import LOSSES
from .utils import weighted_loss
from copy import deepcopy


def xy_wh_r_2_xy_sigma(xywhr):
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.pow(2)).bmm(R.permute(0, 2, 1)).reshape(
        _shape[:-1] + (2, 2))

    return xy, sigma


def xy_stddev_pearson_2_xy_sigma(xy_stddev_pearson):
    _shape = xy_stddev_pearson.shape
    assert _shape[-1] == 5
    xy = xy_stddev_pearson[..., :2]
    stddev = xy_stddev_pearson[..., 2:4]
    pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
    covar = pearson * stddev.prod(dim=-1)
    var = stddev.pow(2)
    sigma = torch.stack((var[..., 0],
                         covar,
                         covar,
                         var[..., 1]), dim=-1).reshape(_shape[:-1] + (2, 2))
    return xy, sigma


def postprocess(distance, fun='log1p', tau=1.0):
    if fun == 'log1p':
        distance = torch.log1p(distance)
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        return 1 - 1 / (tau + distance)
    else:
        return distance


@weighted_loss
def gwd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """
    given any positive-definite symmetrical 2*2 matrix Z:
    Tr(Z^(1/2)) = sqrt(λ_1) + sqrt(λ_2)
    where λ_1 and λ_2 are the eigen values of Z

    meanwhile we have:
    Tr(Z) = λ_1 + λ_2
    det(Z) = λ_1 * λ_2

    combination with following formula:
    (sqrt(λ_1) + sqrt(λ_2))^2 = λ_1 + λ_2 + 2 * sqrt(λ_1 * λ_2)

    yield:
    Tr(Z^(1/2)) = sqrt(Tr(Z) + 2 * sqrt(det(Z)))

    for gwd loss the frustrating coupling part is:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))

    assuming Z = Σp^(1/2) * Σt * Σp^(1/2) then:
    Tr(Z) = Tr(Σp^(1/2) * Σt * Σp^(1/2))
    = Tr(Σp^(1/2) * Σp^(1/2) * Σt)
    = Tr(Σp * Σt)
    det(Z) = det(Σp^(1/2) * Σt * Σp^(1/2))
    = det(Σp^(1/2)) * det(Σt) * det(Σp^(1/2))
    = det(Σp * Σt)

    and thus we can rewrite the coupling part as:
    Tr((Σp^(1/2) * Σt * Σp^(1/2))^(1/2))
    = Tr{Z^(1/2)} = sqrt(Tr(Z) + 2 * sqrt(det(Z)))
    = sqrt(Tr(Σp * Σt) + 2 * sqrt(det(Σp * Σt)))
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    xy_distance = (xy_p - xy_t).pow(2).sum(dim=-1)

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(dim1=-2, dim2=-1).sum(
        dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(0).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt())

    distance = (xy_distance + alpha * alpha * whr_distance).clamp(0).sqrt()

    if normalize:
        scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    # todo
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                               -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                              dim=-1).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(
        dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(
        Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = Sigma_p.det().log()
    Sigma_t_det_log = Sigma_t.det().log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1
    distance = (xy_distance / (alpha * alpha) + whr_distance)
    if sqrt:
        distance = distance.clamp(0).sqrt()

    distance = distance.reshape(_shape[:-1])

    return postprocess(distance, fun=fun, tau=tau)


@weighted_loss
def jd_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    jd = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=False,
                  reduction='none')
    jd = jd + kld_loss(target, pred, fun='none', tau=0, alpha=alpha,
                       sqrt=False,
                       reduction='none')
    jd = jd * 0.5
    if sqrt:
        jd = jd.clamp(0).sqrt()
    return postprocess(jd, fun=fun, tau=tau)


@weighted_loss
def kld_symmax_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    kld_pt = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_tp = kld_loss(target, pred, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_symmax = torch.max(kld_pt, kld_tp)
    return postprocess(kld_symmax, fun=fun, tau=tau)


@weighted_loss
def kld_symmin_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    kld_pt = kld_loss(pred, target, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_tp = kld_loss(target, pred, fun='none', tau=0, alpha=alpha, sqrt=sqrt,
                      reduction='none')
    kld_symmin = torch.min(kld_pt, kld_tp)
    return postprocess(kld_symmin, fun=fun, tau=tau)


@LOSSES.register_module()
class GDLoss(nn.Module):
    BAG_GD_LOSS = {'gwd': gwd_loss,
                   'kld': kld_loss,
                   'jd': jd_loss,
                   'kld_symmax': kld_symmax_loss,
                   'kld_symmin': kld_symmin_loss}
    BAG_PREP = {'xy_stddev_pearson': xy_stddev_pearson_2_xy_sigma,
                'xy_wh_r': xy_wh_r_2_xy_sigma}

    def __init__(self, loss_type, representation='xy_stddev_pearson',
                 fun='log1p', tau=0.0, alpha=1.0, reduction='mean',
                 loss_weight=1.0, **kwargs):
        super(GDLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'none']
        assert loss_type in self.BAG_GD_LOSS
        self.loss = self.BAG_GD_LOSS[loss_type]
        self.preprocess = self.BAG_PREP[representation]
        self.fun = fun
        self.tau = tau
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight >= 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred,
            target,
            fun=self.fun,
            tau=self.tau,
            alpha=self.alpha,
            weight=weight,
            avg_factor=avg_factor,
            reduction=reduction,
            **_kwargs) * self.loss_weight
