import math
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import multivariate_normal
from joblib import Parallel, delayed
from model.Models import get_non_pad_mask, get_end_label
# torch.set_default_dtype(torch.float64)







class SMSTPP(nn.Module):
    def __init__(
        self,
        model,
        T,
        S,
        estimator,
        transformer = None,
        device = "cpu",
        alpha_s = .5, # spatial loss weight
        alpha_CE = 1., # survival loss weight
        grid_t = 5,
        grid_s = 5,
        num_types = 1,
        num_noise=10, 
        sigma_t=0.05, 
        sigma_s=0.05,
        noise_type='normal',
        with_survival=True,
        spatial_weight_looser_factor=0.0,
        eval_grid_t=25,
        eval_grid_s=10,
        intensity_grid_t=25,
        intensity_grid_s=20,
        identity_weight=0

    ):
        super(SMSTPP, self).__init__()
        self.model = model
        self.T = T
        self.S = S
        self.alpha_CE = alpha_CE
        self.alpha_s = alpha_s
        self.estimator = estimator
        self.is_marked = num_types > 1
        self.num_types = num_types
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.transformer = transformer
        self.grid_t = grid_t # grid of t for mle training
        self.grid_s = grid_s # grid of s for mle training
        self.device = device

        self.num_noise = num_noise
        self.sigma_t = sigma_t
        self.sigma_s = sigma_s
        self.noise_type = noise_type
        self.with_survival = with_survival
        self.spatial_weight_looser_factor = spatial_weight_looser_factor
        self.eval_grid_t = eval_grid_t
        self.eval_grid_s = eval_grid_s
        self.intensity_grid_t = intensity_grid_t
        self.intensity_grid_s = intensity_grid_s
        self.identity_weight = identity_weight
    
    def depadding(self, x, mask):
        # This function flatten x and drop the padding
        # x: (batch_size, seq_len, dim) or (batch_size, seq_len)
        # mask: (batch_size, seq_len)

        mask_flat= mask.long().bool().reshape(-1)
        if len(x.shape) == 3:
            x_flat = x.reshape(-1, x.shape[-1])
        else:
            x_flat = x.reshape(-1, 1)
        x_non_mask = x_flat[mask_flat]
        return x_non_mask.unsqueeze(1)

    
    def batch_to_weight(self, event_time, event_loc, mask):
        # event_time: (bs, max_seq_len), true timestamps
        
        device = self.device
        T = self. T; S = self.S

        event_time_origin = torch.concat((torch.zeros(event_time.shape[0],1).to(device), event_time[:,:-1]), dim=1) # (bs, seq_len)
        weight_t = (T - event_time_origin)/2 - abs(event_time - (T + event_time_origin)/2)
        weight_t_grad = torch.where(event_time > (T + event_time_origin)/2, -1, 1)

        # spatial weight
        a1, b1 = S[0]
        a2, b2 = S[1] 
        a2, b2 = a2 * (1 + self.spatial_weight_looser_factor), b2 * (1 + self.spatial_weight_looser_factor)
        a1, b1 = a1 * (1 + self.spatial_weight_looser_factor), b1 * (1 + self.spatial_weight_looser_factor)
        x0 = event_loc[..., 0]
        x1 = event_loc[..., 1]

        c0 = (a1 + b1) * 0.5
        c1 = (a2 + b2) * 0.5
        h0 = (b1 - a1) * 0.5
        h1 = (b2 - a2) * 0.5

        dist_x = h0 - torch.abs(x0 - c0)   
        dist_y = h1 - torch.abs(x1 - c1)   

        weight_s = torch.minimum(dist_x, dist_y).unsqueeze(-1)  # (n, seq_len, 1)
        weight_s_grad = torch.zeros_like(event_loc)  # (n, seq_len, 2)
        mask_x = dist_x < dist_y
        mask_y = ~mask_x

        weight_s_grad[..., 0][mask_x] = -torch.sign(x0[mask_x] - c0)  # d/dx
        weight_s_grad[..., 1][mask_x] = 0.0

        weight_s_grad[..., 0][mask_y] = 0.0
        weight_s_grad[..., 1][mask_y] = -torch.sign(x1[mask_y] - c1)  # d/dy

        # for debugging
        if self.identity_weight:
            weight_s = torch.ones_like(weight_s)
            weight_s_grad = torch.zeros_like(weight_s_grad)

        weight_t_non_mask      = self.depadding(weight_t, mask)
        weight_s_non_mask      = self.depadding(weight_s, mask)
        weight_t_grad_non_mask = self.depadding(weight_t_grad, mask)
        weight_s_grad_non_mask = self.depadding(weight_s_grad, mask)


        return weight_t_non_mask, weight_t_grad_non_mask, weight_s_non_mask, weight_s_grad_non_mask



    
    def get_surviving_loss(self, cond_nonmask_expand, end_mask_nonmask):
        logit = self.model.get_ending_logit(cond_nonmask_expand)

        CE_loss = self.ce_loss(logit, end_mask_nonmask.squeeze().long())

        return CE_loss

    def get_intensity(self, s, t, cond):
        return self.model.get_intensity(s, t, cond, self.with_survival, self.intensity_grid_t, self.intensity_grid_s)


    def wsm_loss(self, event_loc_nonmask, time_gap_nonmask, cond_nonmask,
                                 weight_t_nonmask, weight_t_grad_nonmask, weight_s_nonmask, weight_s_grad_nonmask):
        score_s_nonmask, score_s_grad_nonmask, score_t_nonmask, score_t_grad_nonmask = self.model.get_score(event_loc_nonmask, time_gap_nonmask, cond_nonmask)

        s_loss = (0.5 * score_s_nonmask ** 2 * weight_s_nonmask + score_s_grad_nonmask *weight_s_nonmask + score_s_nonmask * weight_s_grad_nonmask).sum(-1)
        t_loss = (0.5 * score_t_nonmask ** 2 * weight_t_nonmask + score_t_grad_nonmask *weight_t_nonmask + score_t_nonmask * weight_t_grad_nonmask).sum(-1)
        # logit = self.model.get_ending_logit(cond_nonmask_expand)

        # CE_loss = self.ce_loss(logit, end_mask_nonmask.squeeze().long())
        # loss = t_loss.squeeze() + self.alpha_s * s_loss.squeeze() + self.alpha_CE * CE_loss
        return t_loss, s_loss

    def dsm_loss(self, event_loc_nonmask, time_gap_nonmask, cond_nonmask):
        num_noise = self.num_noise
        sigma_t = self.sigma_t
        sigma_s = self.sigma_s
        # if self.noise_type == 'normal':
        t_eps = torch.randn(time_gap_nonmask.shape[0], num_noise, time_gap_nonmask.shape[-1]).to(self.device)
        s_eps = torch.randn(event_loc_nonmask.shape[0], num_noise, event_loc_nonmask.shape[-1]).to(self.device)
        # elif self.noise_type == "lognormal":
            # t_eps = torch.randn(time_gap_nonmask.shape[0], num_noise, time_gap_nonmask.shape[-1]).to(self.device)
            # s_eps = torch.randn(event_loc_nonmask.shape[0], num_noise, event_loc_nonmask.shape[-1]).to(self.device)

        if self.noise_type == "normal":
            t_var = time_gap_nonmask + t_eps * sigma_t
            target_t = - (t_eps) / sigma_t
        elif self.noise_type == "lognormal":
            t_var = time_gap_nonmask * torch.exp(t_eps * sigma_t) 
            target_t = - 1 / (t_var + 1e-10) * (1 + t_eps / sigma_t)
            
        s_var = event_loc_nonmask + s_eps * sigma_s
        score_s_nonmask, _, score_t_nonmask, _ = self.model.get_score(s_var, t_var, cond_nonmask, second_order=False)
        
        target_s = - (s_eps) / sigma_s
        if self.noise_type == "normal":
            t_loss = 0.5 * ((score_t_nonmask - target_t) ** 2).sum() / num_noise * sigma_t ** 2
        elif self.noise_type == "lognormal":
            # t_loss = 0.5 * ((score_t_nonmask - target_t) ** 2).sum() / num_noise / ((2 * sigma_t ** 2) * (1 + 1 / sigma_t ** 2))
            t_loss = 0.5 * ((score_t_nonmask - target_t) ** 2 * sigma_t ** 2 * t_var.detach() ** 2).sum() / num_noise 
        s_loss = 0.5 * ((score_s_nonmask - target_s) ** 2).sum() / num_noise * sigma_s ** 2
        return t_loss, s_loss
    

    def get_cond(self, event_time,  event_loc, non_pad_mask, end_mask, event_mark = None):
        # given N points, this function returns \mathcal F_{0:N} (not \mathcal F_{0:N-1})
        device = self.device
        event_loc_expand = torch.cat((torch.zeros(event_loc.shape[0],1,event_loc.shape[2]).to(device), event_loc), dim=1) # (bs, seq_len, dim)
        event_time_expand = torch.cat((torch.zeros(event_time.shape[0],1).to(device), event_time), dim=1) # (bs, seq_len)
        ############################################################
        if event_mark == None:

            cond_expand, _ = self.transformer(event_loc_expand, event_time_expand)

        else:
            event_mark_expand = torch.cat((torch.zeros(event_mark.shape[0],1).to(device), event_mark), dim=1)
            loc_mark = torch.cat((event_loc_expand, event_mark_expand.unsqueeze(dim=2)), dim=-1)
            cond_expand, _ = self.transformer(loc_mark, event_time_expand)

        non_pad_mask_expand = torch.cat((torch.ones(non_pad_mask.shape[0],1,1).to(device), non_pad_mask), dim=1) # (bs, seq_len, 1+dim)
        cond_nonmask_expand = self.depadding(cond_expand, non_pad_mask_expand)
        end_mask_nonmask = self.depadding(end_mask, non_pad_mask_expand)
        keep = (end_mask_nonmask==0).squeeze()
        cond_nonmask = cond_nonmask_expand[keep]
        
        return cond_nonmask_expand, cond_nonmask
        


    
    def losses(self, x, weight_t, weight_t_grad, weight_s, weight_s_grad, cond=None):
        score_t, score_t_grad, score_s, score_s_grad = self.model.get_score(x, cond)
        t_loss = 0.5 * score_t ** 2 * weight_t + score_t_grad *weight_t + score_t * weight_t_grad
        s_loss = (0.5 * score_s ** 2 * weight_s + score_s_grad *weight_s + score_s * weight_s_grad).sum(-1)
        loss = t_loss.squeeze() + s_loss.squeeze()
        return loss
    

    def forward(self, batch, mode):

        device = self.device
        if self.num_types == 1:
            event_time_origin, time_gap, lng, lat = map(lambda x: x.to(device), batch)
            event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)
        else:
            event_time_origin, time_gap, mark, lng, lat = map(lambda x: x.to(device), batch)
            event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)
    

        event_time = event_time_origin + time_gap
        non_pad_mask = get_non_pad_mask(event_time)
        non_pad_mask_expand = torch.cat((torch.ones(non_pad_mask.shape[0],1,1).to(device), non_pad_mask), dim=1) # (bs, seq_len, 1+dim)
        end_mask = get_end_label(non_pad_mask.squeeze())

        if not self.with_survival:
            event_time_end = torch.concat((event_time, torch.zeros(event_time.shape[0],1).to(device)), dim=1)
            non_pad_mask_temp = torch.concat((non_pad_mask, torch.zeros(event_time.shape[0],1,1).to(device)), dim=1)
            event_time_end = event_time_end + (non_pad_mask_expand - non_pad_mask_temp).squeeze() *self.T
            time_gap_end = torch.concat((event_time_end[:,0:1], event_time_end[:,1:] - event_time_end[:,:-1]), dim=1) * non_pad_mask_expand.squeeze()
            time_gap_end_nonmask = self.depadding(time_gap_end, non_pad_mask_expand)
        else:
            time_gap_end_nonmask = None

        event_loc_nonmask = self.depadding(event_loc, non_pad_mask)
        event_time_nonmask = self.depadding(event_time, non_pad_mask)
        time_gap_nonmask = self.depadding(time_gap, non_pad_mask)
        end_mask_nonmask = self.depadding(end_mask, non_pad_mask_expand)

        # forward the transformer
        if self.num_types == 1:
            cond_nonmask_expand, cond_nonmask = self.get_cond(event_time, event_loc, non_pad_mask, end_mask)
        else:
            cond_nonmask_expand, cond_nonmask = self.get_cond(event_time, event_loc, non_pad_mask, end_mask, event_mark=mark)


        end_mask_nonmask = end_mask_nonmask.bool()
    
        # print(cond_nonmask[0])


        bs = event_time.shape[0]
        if self.num_types > 1:
            mark_nonmask = self.depadding(mark, non_pad_mask)
        if mode == "train":
            if self.estimator == "wsm":
                weight_t_nonmask, weight_t_grad_nonmask, weight_s_nonmask, weight_s_grad_nonmask = self.batch_to_weight(event_time, event_loc, non_pad_mask)

                t_loss, s_loss = self.wsm_loss(event_loc_nonmask, time_gap_nonmask, cond_nonmask,
                                    weight_t_nonmask, weight_t_grad_nonmask, weight_s_nonmask, weight_s_grad_nonmask)
                loss = t_loss.squeeze().sum() + self.alpha_s * s_loss.squeeze().sum()
                if self.with_survival:
                    CE_loss = self.get_surviving_loss(cond_nonmask_expand, end_mask_nonmask)
                    loss += CE_loss.sum()
                loss /= bs
                if self.num_types > 1:
                    label_loss = self.model.get_label_ll(time_gap_nonmask, event_loc_nonmask, cond_nonmask, mark_nonmask)
                    loss -= label_loss.sum() / bs
            elif self.estimator == "mle":
                if self.num_types == 1:
                    tll, sll = self.model.get_lls(event_loc_nonmask, event_time_nonmask, time_gap_nonmask, cond_nonmask_expand, cond_nonmask, end_mask_nonmask, num_grid=self.grid_t, num_grid_s=self.grid_s, with_survival=self.with_survival, t_expand=time_gap_end_nonmask)
                    loss = - (tll + sll) / bs
                    # debug
                    # loss = - tll / bs
                else:
                    tll, sll, label_ll = self.model.get_lls(event_loc_nonmask, event_time_nonmask, time_gap_nonmask, cond_nonmask_expand, cond_nonmask, end_mask_nonmask, \
                        num_grid=self.grid_t, num_grid_s=self.grid_s, mark = mark_nonmask, with_survival=self.with_survival, t_expand=time_gap_end_nonmask)
                    loss = - (tll + sll + label_ll) / bs
            elif self.estimator == "dsm":
                t_loss, s_loss = self.dsm_loss(event_loc_nonmask, time_gap_nonmask, cond_nonmask)
                loss = t_loss + s_loss
                if self.with_survival:
                    CE_loss = self.get_surviving_loss(cond_nonmask_expand, end_mask_nonmask)
                    loss += self.alpha_CE * CE_loss.sum()
                loss /= bs
                # loss = (t_loss + s_loss + self.alpha_CE * CE_loss.sum()) / bs
                if self.num_types > 1:
                    label_loss = self.model.get_label_ll(time_gap_nonmask, event_loc_nonmask, cond_nonmask, mark_nonmask)
                    loss -= label_loss.sum() / bs            

            return loss, bs
        
        elif mode == "test":
            if self.num_types == 1:
                t_ll, s_ll = self.model.get_lls(event_loc_nonmask, event_time_nonmask, time_gap_nonmask, cond_nonmask_expand, cond_nonmask, end_mask_nonmask,\
                     num_grid=self.eval_grid_t, num_grid_s=self.eval_grid_s, with_survival=self.with_survival, t_expand=time_gap_end_nonmask)
                return t_ll, s_ll, bs
                # return t_ll, torch.tensor(0.), bs
            else:
                t_ll, s_ll, label_ll = self.model.get_lls(event_loc_nonmask, event_time_nonmask, time_gap_nonmask, cond_nonmask_expand, cond_nonmask, end_mask_nonmask, \
                    num_grid=self.eval_grid_t, num_grid_s=self.eval_grid_s, mark = mark_nonmask, with_survival=self.with_survival, t_expand=time_gap_end_nonmask)

                label_logit = self.model.get_label_logits(time_gap_nonmask, event_loc_nonmask, cond_nonmask)
                label_pred = torch.argmax(label_logit, dim=-1)
                label_acc = (label_pred.squeeze() == mark_nonmask.squeeze()).sum() / bs
                return t_ll, s_ll, label_ll, bs, label_acc


  
