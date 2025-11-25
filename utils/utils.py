import model.Constants as Constants
from model.Layers import EncoderLayer
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.plot_map import plot_density
from utils.synthetic_stpp import AutoIntGaussianHawkes, HawkesLam, AutoIntGaussianSelfCorrecting


def depadding(x, mask):
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

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_end_label(mask):
    # mask: (batch_size, seq_len)
    # This function returns a nonmask label, if 1 represents this is the end of the sequence
    
    # First, we calculate the row sum of the mask
    # if mask is of shape (seq_len,), we need to reshape it to (1, seq_len)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    device = mask.device
    row_sum = mask.sum(dim=1).squeeze().to(int)  # (batch_size)
    end_mask = torch.zeros(mask.shape[0], mask.shape[1] + 1, 1).to(device)
    end_mask[torch.arange(end_mask.size(0), device=end_mask.device), row_sum , 0] = 1 # (batch_size, seq_len)
    return end_mask



def test_lamb_temporal(testloader, model, device, name, gt_model, resolution=10):
    from model.Models import get_non_pad_mask, get_end_label
    eval_event = []
    max_len = 0
    for batch in testloader:
        # get batch data
        event_time_origin, time_gap, lng, lat = map(lambda x: x.to(device), batch)
        event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)
        event_time = event_time_origin + time_gap
        non_pad_mask = get_non_pad_mask(event_time)
        row_sum = non_pad_mask.sum(dim=1)
        idx = torch.argmax(row_sum)
        if row_sum[idx] > max_len:
            max_len = row_sum[idx]
            eval_event = [event_loc[idx],event_time[idx], time_gap[idx]]
        # break
    
    eval_loc = eval_event[0].unsqueeze(0)
    eval_time = eval_event[1].unsqueeze(0)
    eval_time_gap = eval_event[2].unsqueeze(0)
    non_pad_mask = get_non_pad_mask(eval_time)
    end_mask = get_end_label(non_pad_mask.squeeze())
    _, cond = model.get_cond(eval_time, eval_loc, non_pad_mask, end_mask)

    # evaluate the temporal intensity
    t_resol =  (torch.linspace(0, 0.99, resolution, device=device)\
            .unsqueeze(0) * eval_time_gap.squeeze().unsqueeze(-1)).unsqueeze(-1)
    eval_time_prior = torch.concat((torch.zeros(1,).to(device), eval_time.squeeze()[:-1]))
    time_resol = eval_time_prior.unsqueeze(-1) + t_resol.squeeze(-1)
    cond_resol = cond.repeat(1, resolution, 1)
    intensity_resol = model.model.get_tilde_intensity_t(t_resol, cond_resol).squeeze()

    #eval true intensity
    # name is estimator_dataset_model_seed
    # extract dataseet from name

    plt.figure()
    plt.plot(time_resol.flatten().detach().cpu().numpy(), intensity_resol.flatten().detach().cpu().numpy(), label="Predicted Intensity")
    
    dataset = name.split('_')[1]
    if dataset in ['AutoIntHawkes', 'AutoIntHawkes2','GaussianHawkesKernelFactorized','AutoIntTest','AutoIntHawkes3', 'AutoIntHawkes4']:
        intensity_gt_resol = torch.zeros_like(intensity_resol.squeeze())
        hist_t = []
        hist_s = []
        for i in range(eval_time.squeeze().shape[0]):
            for j in range(resolution):
                intensity_gt_resol[i, j] = gt_model.kernel.lambda_T(time_resol[i,j].detach().cpu().numpy(), hist_t, hist_s)
            hist_t.append(eval_time.squeeze()[i].detach().cpu().numpy())
            hist_s.append(eval_loc.squeeze()[i].detach().cpu().numpy())
    

        plt.plot(time_resol.flatten().detach().cpu().numpy(), intensity_gt_resol.flatten().detach().cpu().numpy(), label="True Intensity")

    # plot the realized timestamps
    plt.scatter(eval_time.squeeze().detach().cpu().numpy(), np.full_like(eval_time.squeeze().detach().cpu().numpy(), -0.8), label="Event", marker='x')
    plt.legend()
    plt.savefig(f"./image/intensity_gt_vs_pred_{name}.pdf")
    plt.close()




def test_lamb_spatial(testloader, trained_models, device, names, dataset,gt_model, S, T_max):
    from utils.autoint_plotter import plot_lambst_interactive, plot_lambst_panels
    # iterate over testloader
    from model.Models import get_non_pad_mask, get_end_label
    # eval_event = []
    # max_len = 0
    for idx_loader ,batch in enumerate(testloader):
        # get batch data
        event_time_origin, time_gap, lng, lat = map(lambda x: x.to(device), batch)
        event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)
        event_time = event_time_origin + time_gap
        # eval_event=([event_loc[0],event_time[0], time_gap[0]])
        non_pad_mask = get_non_pad_mask(event_time)
        row_sum = non_pad_mask.sum(dim=1)
        idx = torch.argmax(row_sum)
        # names is a list of names
        # we concat all names in names
        name_ = dataset + "_" + "_".join(names) + f"_{idx_loader}"
        # name_ = name +f"_{idx_loader}"
        max_len = row_sum[idx]
        eval_loc = event_loc[idx].unsqueeze(0)
        eval_time = event_time[idx].unsqueeze(0)
        eval_time_gap = time_gap[idx].unsqueeze(0)

    
        # evaluate the spatial intensity
        nx = ny = 128
        xs = torch.linspace(S[0][0], S[0][1], nx, device=device)
        ys = torch.linspace(S[1][0], S[1][1], ny, device=device)
        YY, XX = torch.meshgrid(ys, xs, indexing="ij")                # (ny,nx)
        grid = torch.stack([XX, YY], dim=-1).reshape(-1, 2) 
        
        # we choose four slice to evaluate the intensity
        pre_idx = 3
        timestamps = [pre_idx, int(1/4*max_len)+pre_idx, int(1/2*max_len)+pre_idx, int(3/4*max_len)+pre_idx]
        if 3/4*max_len+pre_idx+1 > max_len:
            raise ValueError("max_len is too short, change pre_idx")
        lams = np.zeros((len(trained_models), len(timestamps), ny, nx))

        for model_idx, model in enumerate(trained_models):
            non_pad_mask = get_non_pad_mask(eval_time)

            end_mask = get_end_label(non_pad_mask.squeeze())
            _, cond = model.get_cond(eval_time, eval_loc, non_pad_mask, end_mask)
            eval_time_gap_nonmask = depadding(eval_time_gap, non_pad_mask)
            for timestamp_idx, timestamp in enumerate(timestamps):
                # so we use H_{timestamp} and predict the next event
                cond_ = cond[timestamp:timestamp + 1]
                time_ = eval_time_gap_nonmask[timestamp:timestamp + 1]
                
                # evaluate the intensity
                with torch.no_grad():
                    B = 2048
                    outs = []
                    for i in range(0, grid.shape[0], B):
                        g = grid[i:i+B].unsqueeze(-2)
                        cond_aug = cond_.repeat(1, B, 1).reshape((-1, cond_.shape[1], cond_.shape[2]))
                        time_aug = time_.repeat(1, B, 1).reshape((-1, time_.shape[1], time_.shape[2]))
                        val = model.get_intensity(g, time_aug, cond_aug).squeeze() 
                        outs.append(val)
                    v = torch.stack(outs)
                    Z = v.reshape(ny, nx).clamp_min(0)

                Z = Z.detach().cpu().numpy()
                lams[model_idx, timestamp_idx, :, :] = Z
        
        if gt_model is not None:

            gt_lams = np.zeros((1, len(timestamps), ny, nx))
            # reshape grid to (ny, nx, 2)
            grid = grid.reshape(ny, nx, 2)
            for timestamp_idx, timestamp in enumerate(timestamps):
                # calculate ground truth lambda
                hist_t = eval_time.squeeze().detach().cpu().numpy()[:timestamp-1]
                hist_s = eval_loc.squeeze().detach().cpu().numpy()[:timestamp-1]
                t = eval_time.squeeze().detach().cpu().numpy()[timestamp]
                # calculate intensity on grids
                for i in range(ny):
                    for j in range(nx):
                        s = grid[i, j].detach().cpu().numpy()
                        lamb = gt_model.value(t, hist_t, s, hist_s)
                        gt_lams[0, timestamp_idx, i, j] = lamb
            all_lambs = np.concatenate([gt_lams, lams], axis=0)
        else:
            all_lambs = lams

        t_range = eval_time.squeeze()[timestamps].detach().cpu().numpy()

        cmax = np.max(all_lambs)*1.1
        # plot_lambst_interactive(all_lambs, xs.detach().cpu().numpy(), ys.detach().cpu().numpy(), t_range, cmin=0, cmax=cmax, scaler=None, heatmap=False, show=True, cauto=False, master_title='Spatio-temporal Conditional Intensity', subplot_titles=['Model', 'Ground Truth'],
        # name=name_)
        savepath = f"./htmls/intensity_gt_vs_pred_{name_}.pdf"

        history_s = eval_loc.squeeze().detach().cpu().numpy()[:timestamps[-1]]
        history_t = eval_time.squeeze().detach().cpu().numpy()[:timestamps[-1]]
        history = (history_s, history_t)
        if gt_model is not None:
            model_names = ['GT'] + names
        else:
            model_names = names
        plot_lambst_panels(all_lambs, T_max, xs.detach().cpu().numpy(), ys.detach().cpu().numpy(), t_range, cmin=0, cmax=cmax, model_names=model_names, savepath=savepath, history=history)




def plot_map_temperature(testloader, model, device, name, Max, Min, n_traj=1):


    def make_loglik_fn(model, cond_, t_snapshot):
        def loglik_fn(S_norm):  # S_norm is the standardized grid points passed to plot_density
            with torch.no_grad():
                # t_snapshot is the same for all grid points
                t_batch = torch.full((S_norm.shape[0], 1),
                                    float(t_snapshot),
                                    device=S_norm.device)
                cond = cond_.repeat(S_norm.shape[0], 1, 1)
                # Here you can change the interface of your model:
                lam = model.get_intensity(
                    s=S_norm.unsqueeze(1),
                    t=t_batch.unsqueeze(1),
                    cond=cond,
                )  # Expected shape: (num_points,)
                loglam = torch.log(lam + 1e-8)
            return loglam

        return loglik_fn
    
    # take the first sequence in testloader
    for i in range(n_traj):
        seq = testloader.dataset.__getitem__(i)
        # seq = testloader[0]
        # seq contains four lists, we need to turn them into tensors
        event_time_origin = torch.tensor(seq[0]).to(device).unsqueeze(0)
        time_gap = torch.tensor(seq[1]).to(device).unsqueeze(0)
        lng = torch.tensor(seq[2]).to(device).unsqueeze(0)
        lat = torch.tensor(seq[3]).to(device).unsqueeze(0)
        event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)
        event_time = event_time_origin + time_gap
        non_pad_mask = get_non_pad_mask(event_time)
        end_mask = get_end_label(non_pad_mask.squeeze())
        _, cond = model.get_cond(event_time, event_loc, non_pad_mask, end_mask)

        len_ = event_time.shape[1]
        pre_idx = 3
        timestamps = [pre_idx, int(1/4*len_)+pre_idx, int(1/2*len_)+pre_idx, int(3/4*len_)+pre_idx, len_-2]
        if 3/4*len_+pre_idx+1 > len_ - 2:
            raise ValueError("len_ is too short, change pre_idx")

        # get dataset name from name
        dataset_name = name.split('_')[0]
        
        if dataset_name not in ['Earthquake', 'COVID19', 'Citibike']:
            raise ValueError("Dataset not supported")
        else:
            if dataset_name == 'Earthquake':
                dataset_name = 'earthquakes_jp'
            elif dataset_name == 'COVID19':
                dataset_name = 'covid_nj_cases'
            elif dataset_name == 'Citibike':
                dataset_name = 'citibike'
        
        mean= torch.tensor(Min[1:3])
        std = torch.tensor(Max[1:3]) - mean

        name_i = name + f"_{i}"
        for timestamp in timestamps:
            cond_ = cond[timestamp:timestamp + 1]
            time = time_gap.squeeze()[timestamp]
            loc = event_loc.squeeze()[:timestamp]
            loglik_fn = make_loglik_fn(model, cond_, time)
            savepath = f"./image"
            plot_density(loglik_fn, loc, timestamp, mean, std, savepath, dataset_name, device, text=None, fp64=False, estimator_name=name_i)






def choose_kernel(dataset_name, S):
    if dataset_name not in ['AutoIntHawkes2','AutoIntHawkes3','SC1','SC4']:
        return None
    elif dataset_name == 'AutoIntHawkes2':
        kernel = AutoIntGaussianHawkes(
                mu=.5,
                alpha=0.8,
                beta=1.0,
                sigma0_x=.2,
                sigma0_y=.2,
                sigma2_x=0.5,
                sigma2_y=0.5,
                center0=(0.5, 0.5),
                S=S
        )
    elif dataset_name == 'AutoIntHawkes3':
        kernel = AutoIntGaussianHawkes(
                mu=1.,
                alpha=1.,
                beta=2.0,
                sigma0_x=.2,
                sigma0_y=.2,
                sigma2_x=0.1,
                sigma2_y=0.1,
                center0=(0.5, 0.5),
                S=S
        )
    elif dataset_name == 'SC1':
        kernel = AutoIntGaussianSelfCorrecting(
                mu=1.,
                alpha=0.4,
                beta=0.2,
                sigma0_x=0.25,
                sigma0_y=0.25,
                sigma2_x=0.2,
                sigma2_y=0.2,
                center0=(0.5, 0.5),
                S=S
        )
    elif dataset_name == 'SC4':
        kernel = AutoIntGaussianSelfCorrecting(
                mu=1.,
                alpha=0.2,
                beta=0.2,
                sigma0_x=1,
                sigma0_y=1,
                sigma2_x=0.85,
                sigma2_y=0.85,
                center0=(0.5, 0.5),
                S=S
        )
    
    gt_model = HawkesLam(0, kernel, maximum=kernel.max_intensity())
    return gt_model

