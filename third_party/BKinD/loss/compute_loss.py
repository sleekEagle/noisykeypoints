## Sum all losses here
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.separation_loss import separation_loss
from loss.rotation_loss import rotation_loss
from loss.ssim_loss import compute_ssim
from loss.change_target import change_target, coarse_target  # noisykeypoints
from model.scale_sigma import multi_scale_args, multi_scale_normalize  # noisykeypoints

from model.vggforLoss import mse_loss_mask, VGG_feat

def define_loss(args):
    loss_model = VGG_feat()
    loss_model = torch.nn.DataParallel(loss_model).cuda(args.gpu)
    loss_model = loss_model.eval()

    recon_crit = mse_loss_mask().cuda(args.gpu)
    separation_crit = separation_loss(args.nkpts).cuda(args.gpu)
    rotation_crit = rotation_loss().cuda(args.gpu)

    criterion = [recon_crit, nn.MSELoss().cuda(args.gpu), separation_crit, rotation_crit]

    return loss_model, criterion

class computeLoss:
    def __init__(self, args, model=None):
        self.args = args
        self.model = model  # noisykeypoints: multi_scale reads model.last_log_var

        self.loss_model = VGG_feat()
        self.loss_model = torch.nn.DataParallel(self.loss_model).cuda(args.gpu)
        self.loss_model = self.loss_model.eval()

        recon_crit = mse_loss_mask().cuda(args.gpu)
        separation_crit = separation_loss(args.nkpts).cuda(args.gpu)
        rotation_crit = rotation_loss().cuda(args.gpu)

        self.criterion = [recon_crit, nn.MSELoss().cuda(args.gpu), separation_crit, rotation_crit]

    def _recon_loss(self, to_pred, recon, loss_mask, amp):
        """BKinD's reconstruction loss (pixels + VGG features, each divided by its perc_weight),
        per image: (B,). Its batch mean is BKinD's original value."""
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            # noisykeypoints: the target is a constant and VGG is not trained, so its features need no
            # gradient graph (same values, less memory; matters with several scale heads)
            with torch.no_grad():
                vgg_feat_in = self.loss_model(to_pred)
            vgg_feat_out = self.loss_model(recon)
        vgg_feat_in = _to_float(vgg_feat_in)
        vgg_feat_out = _to_float(vgg_feat_out)

        l = self.criterion[0](to_pred, recon, loss_mask)
        wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[0])
        l /= wl
        loss = l.mean((1, 2, 3))

        for _i in range(0, len(vgg_feat_in)):
            _mask = F.upsample(loss_mask,
                               size=(vgg_feat_in[_i].size(2), vgg_feat_in[_i].size(3)), mode='bilinear') # in_mask
            l = self.criterion[0](vgg_feat_in[_i], vgg_feat_out[_i], _mask)
            wl = _exp_running_avg(l.mean(), init_val=self.args.perc_weight[_i+1])
            l /= wl
            loss = loss + l.mean((1, 2, 3))
        return loss

    def update_loss(self, inputs, tr_inputs, loss_mask, output, epoch, tracks=None, means=None):
        # noisykeypoints: with mixed precision the model outputs may be float16. All loss
        # maths (SSIM target, MSE, rotation, separation) stays in float32; only the VGG
        # feature network runs in float16, and its features are cast back to float32.
        amp = bool(getattr(self.args, 'amp', False))
        output = _to_float(output)
        # noisykeypoints: BKinD's SSIM target, or a noise-robust one (loss/change_target.py)
        # (with `temporal`, the target is built from the k-frame means around A and B instead)
        to_pred = change_target(self.args, *(means if means is not None else (inputs, tr_inputs)), tracks)
        self.last_target = to_pred[:, :1].detach()
        if getattr(self.args, 'motion_mask', None):
            # noisykeypoints: background-subtraction mask (dataloader/background.py). BKinD sums the
            # masked error over all pixels, so a mask covering ~10% of the picture would shrink the
            # reconstruction loss ~10x next to the rotation and separation terms. Rescale each image's
            # mask to mean 1 (at most 50x, for pairs where almost nothing is foreground).
            loss_mask = loss_mask.float()
            loss_mask = loss_mask / loss_mask.mean((1, 2, 3), keepdim=True).clamp(min=0.02)

        factors, _ = multi_scale_args(self.args)
        if factors:
            # noisykeypoints: one decoder head per resolution, weighted by learned uncertainty
            # (model/scale_sigma.py). Head k redraws the change target at factors[k].
            if getattr(self.args, 'change_target', None):
                raise ValueError('multi_scale replaces change_target; set only one')
            log_var = self.model.last_log_var.float()                     # (B, n_scales)
            targets = [compute_ssim(inputs.float(), tr_inputs.float()) if f == 1 else
                       coarse_target(inputs, tr_inputs, f) for f in factors]
            L = torch.stack([self._recon_loss(t, output[0][:, 3 * k:3 * k + 3], loss_mask, amp)
                             for k, t in enumerate(targets)], 1)             # (B, n_scales)
            if multi_scale_normalize(self.args):
                # noisykeypoints: total weight fixed at 1 (as in ct4); sigma only splits it between heads.
                # sigma is trained by the uncertainty loss on the DETACHED head losses, so it cannot
                # change how hard the change loss pushes on the keypoints next to rotation/separation.
                p = torch.softmax(-log_var, 1)                                # (B, n_scales), rows sum to 1
                loss = (p.detach() * L).sum(1).mean() +                     (L.detach() * torch.exp(-log_var) + log_var).mean(1).mean()
                weight = p.detach().mean(0)
            else:
                loss = (L * torch.exp(-log_var) + log_var).mean(1).mean()
                weight = torch.exp(-log_var).detach().mean(0)
            self.last_scale = dict(L=L.detach().mean(0), weight=weight, factors=factors)
            self.last_head = int(weight.argmax())                            # head shown in samples
            self.last_target = targets[self.last_head][:, :1].detach()
        else:
            loss = self._recon_loss(to_pred, output[0], loss_mask, amp).mean()

        if epoch >= self.args.curriculum:
            deg = torch.ones((output[2][1].size()[0]), device=output[2][1].device) * 90
            rot_loss, rot_label = self.criterion[3](output[2][0][:, :self.args.nkpts], output[2][1], deg)
            loss += rot_loss/3

            deg = torch.ones((output[2][1].size()[0]), device=output[2][1].device) * 180
            rot_loss, rot_label2 = self.criterion[3](output[2][0][:, :self.args.nkpts], output[2][2], deg)
            loss += rot_loss/3

            deg = torch.ones((output[2][1].size()[0]), device=output[2][1].device) * -90
            rot_loss, rot_label3 = self.criterion[3](output[2][0][:, :self.args.nkpts], output[2][3], deg)
            loss += rot_loss/3

            separation = self.criterion[2]((output[5][0], output[5][1]))
            loss += separation.mean()

            separation = self.criterion[2]((output[5][2], output[5][3]))
            loss += separation.mean()

            separation = self.criterion[2]((output[5][4], output[5][5]))
            loss += separation.mean()

            separation = self.criterion[2](output[1])
            loss += separation.mean()

        return loss


def _to_float(x):
    """noisykeypoints: cast every float tensor in a nested tuple/list to float32."""
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, (tuple, list)):
        return type(x)(_to_float(v) for v in x)
    return x


# x_avg: torch variable which is initialized to init_val - weight
def _exp_running_avg(x, rho=0.99, init_val=0.0):
    x_avg = init_val

    w_update = 1.0 - rho
    x_new = x_avg + w_update * (x - x_avg)
    return x_new
