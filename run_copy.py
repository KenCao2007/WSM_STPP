import torch
import torch.nn as nn
import numpy as np
from model.Models import Transformer_ST, get_non_pad_mask
from model.SM_model import SMSTPP
from model.intensity_model import smash_intensity_score
from torch.optim import AdamW
import argparse
from utils.Dataset import get_dataloader
import time
import datetime
import pickle
import os
from tqdm import tqdm
import random
import math
import matplotlib.pyplot as plt
from utils.utils import test_lamb_spatial, test_lamb_temporal, plot_map_temperature, choose_kernel

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S",TIME)

def normalization(x,MAX,MIN):
    return (x-MIN)/(MAX-MIN)

def denormalization(x,MAX,MIN,log_normalization=False):
    if log_normalization:
        return torch.exp(x.detach().cpu()*(MAX-MIN)+MIN)
    else:
        return x.detach().cpu()*(MAX-MIN)+MIN



def plot_location(data):
    # data: list of all sequences
    sequences = data
    pairs = [(ev[2], ev[3]) for seq in sequences for ev in seq if len(ev) >= 4]
    pos = torch.tensor(pairs, dtype=torch.float32) if pairs else torch.empty(0, 2)

    alpha = (1 - math.sqrt(0.95)) / 2  # ≈ 0.0126603
    x = pos[:, 0]; y = pos[:, 1]

    x_min = torch.quantile(x, alpha).item()
    x_max = torch.quantile(x, 1 - alpha).item()
    y_min = torch.quantile(y, alpha).item()
    y_max = torch.quantile(y, 1 - alpha).item()# 
    Max = [x_max, y_max]
    Min = [x_min, y_min]

    xy = pos.cpu().numpy()
    plt.figure()
    plt.scatter(xy[:, 0], xy[:, 1], s=4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x'); plt.ylabel('y'); plt.title('All points')
    plt.show()
    plt.savefig('./location.png')
    return Max, Min


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1, help='')
    parser.add_argument('--model', type=str, default='SMASH_decouple', help='', choices=['SMASH', 'SMASH_decouple', 'SMASH_GMM'])
    parser.add_argument('--mode', type=str, default='train', help='', choices=['train', 'debug', 'test'])
    parser.add_argument('--total_epochs', type=int, default=300, help='')
    parser.add_argument('--dim', type=int, default=2, help='number of dimensions in a batch of data, for example, if the data is [time, loc_x, loc_y], then dim = 3', choices = [1,2,3,4])
    parser.add_argument('--dataset', type=str, default='Earthquake',choices=['Earthquake','Crime','football','Gaussian',"COVID19","Citibike",'AutoIntHawkes2', 'AutoIntHawkes3', 'SC4', 'SC1'] , help='')
    parser.add_argument('--batch_size', type=int, default=96,help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--save_path', type=str, default='./ModelSave/debug/')
    parser.add_argument('--cond_dim', type=int, default=48, help='')
    parser.add_argument("--estimator", type=str, default="wsm", choices=["wsm", "mle","dsm"])
    parser.add_argument('--normalize_location', type=int, default=1, help='')
    parser.add_argument('--alpha_CE', type=float, default=1.0, help='')
    parser.add_argument('--alpha_s', type=float, default=5., help='')
    parser.add_argument('--grid_t', type=int, default=10, help='')
    parser.add_argument('--grid_s', type=int, default=10, help='')
    parser.add_argument('--num_units', type=int, default=48, help='')
    parser.add_argument('--num_types', type=int, default=1, help='')
    parser.add_argument('--normalize_time', type=int, default=1, help='')
    parser.add_argument('--noise_type', type=str, default='lognormal', choices=[ 'lognormal'], help='currently only support lognormal')
    parser.add_argument('--num_noise', type=int, default=10, help='')
    parser.add_argument('--sigma_t', type=float, default=0.5, help='')
    parser.add_argument('--sigma_s', type=float, default=0.1, help='')
    parser.add_argument('--with_survival', type=int, default=0, help='')
    parser.add_argument('--K_trig', type=int, default=2, help='') # 4 for AutoIntHawkes 3
    parser.add_argument('--spatial_weight_looser_factor', type=float, default=0.0, help='')
    parser.add_argument('--identity_weight', type=int, default=0, help='')
    parser.add_argument('--n_head', type=int, default=4, help='transformer parameter')
    parser.add_argument('--n_layers', type=int, default=4, help='transformer parameter')
    parser.add_argument('--d_k', type=int, default=16, help='transformer parameter')
    parser.add_argument('--d_v', type=int, default=16, help='transformer parameter')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--eval_grid_t', type=int, default=25, help='number of grid points in the time dimension for evaluation')
    parser.add_argument('--eval_grid_s', type=int, default=10, help='number of grid points in the space dimension for evaluation')
    parser.add_argument('--intensity_grid_t', type=int, default=25, help='number of grid points in the time dimension for intensity calculation')
    parser.add_argument('--intensity_grid_s', type=int, default=20, help='number of grid points in the space dimension for intensity calculation')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    return args

opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim=1

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)

def data_loader(opt):

    f = open('./dataset/{}/data_train.pkl'.format(opt.dataset),'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    f = open('dataset/{}/data_val.pkl'.format(opt.dataset),'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    f = open('dataset/{}/data_test.pkl'.format(opt.dataset),'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]

    Max, Min = [], []
    if opt.dataset == 'football':
        opt.num_types = 7
        opt.dim = 3
    for m in range(opt.dim+1):
        Max.append(max([i[m] for u in train_data+test_data+val_data for i in u]))
        Min.append(min([i[m] for u in train_data+test_data+val_data for i in u]))
    print('Min & Max', (Max, Min))
    



    if opt.normalize_time:
        train_data = [[[normalization(i[0], Max[0], Min[0]) ] + i[1:] for i in u] for u in train_data]
        val_data = [[[normalization(i[0], Max[0], Min[0]) ] + i[1:] for i in u] for u in val_data]
        test_data = [[[normalization(i[0], Max[0], Min[0])] + i[1:] for i in u] for u in test_data]

    # normalize the location, do not change the time
    if opt.normalize_location and opt.num_types == 1:
        train_data = [[[normalization(i[j], Max[j], Min[j]) if j > 0 else i[j] for j in range(len(i))]for i in u] for u in train_data]
        val_data = [[[normalization(i[j], Max[j], Min[j]) if j > 0 else i[j] for j in range(len(i))]for i in u] for u in val_data]
        test_data = [[[normalization(i[j], Max[j], Min[j]) if j > 0 else i[j] for j in range(len(i))]for i in u] for u in test_data]
    elif opt.normalize_location and opt.num_types > 1:
        train_data = [[[normalization(i[j], Max[j], Min[j]) if (j > 0 and j != 1) else i[j] for j in range(len(i))]for i in u] for u in train_data]
        val_data = [[[normalization(i[j], Max[j], Min[j]) if (j > 0 and j != 1) else i[j] for j in range(len(i))]for i in u] for u in val_data]
        test_data = [[[normalization(i[j], Max[j], Min[j]) if (j > 0 and j != 1) else i[j] for j in range(len(i))]for i in u] for u in test_data]
    # modify the timez
    train_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]]+ i[2:] for index, i in enumerate(u)] for u in train_data]
    val_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]]+ i[2:] for index, i in enumerate(u)] for u in val_data]
    test_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0], i[1]]+ i[2:] for index, i in enumerate(u)] for u in test_data]

    if opt.dataset == 'football':
        # we delete points that the time_gap is zero
        train_data = [[i for i in u if i[1] != 0] for u in train_data]
        val_data = [[i for i in u if i[1] != 0] for u in val_data]
        test_data = [[i for i in u if i[1] != 0] for u in test_data]

    
    train_data = [[[u[index-1][0] if index>0 else 0]+ i[1:] for index, i in enumerate(u)] for u in train_data]
    val_data = [[[u[index-1][0] if index>0 else 0]+ i[1:] for index, i in enumerate(u)] for u in val_data]
    test_data = [[[u[index-1][0] if index>0 else 0]+ i[1:] for index, i in enumerate(u)] for u in test_data]

    # for each element in data, it takes form [prior_stamp, stamp_diff, now_loc_x, now_loc_y]

    Max_new, Min_new = [], []
    for m in range(opt.dim+2):
        # if m > 1:
        Max_new.append(max([i[m] for u in train_data+test_data+val_data for i in u]))
        Min_new.append(min([i[m] for u in train_data+test_data+val_data for i in u]))


    if not opt.normalize_location and opt.dataset in ['AutoIntHawkes3','AutoIntHawkes2','SC4','SC1']:
        Max_new[2] = 1
        Max_new[3] = 1
        Min_new[2] = 0
        Min_new[3] = 0
    if opt.normalize_location and opt.num_types == 1:
        Max_new[2] = 1
        Max_new[3] = 1
        Min_new[2] = 0
        Min_new[3] = 0
    if opt.normalize_time:
        Max_new[0] = 1
        Min_new[0] = 0

    print('Min & Max after normalization', (Max_new, Min_new))

    plot_location(train_data)

    trainloader = get_dataloader(train_data, opt.batch_size, D = opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, D = opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, opt.batch_size, D = opt.dim, shuffle=False)

    return trainloader, testloader, valloader, (Max,Min), (Max_new, Min_new)

def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num



if __name__ == "__main__":
    
    setup_init(opt)

    print('dataset:{}'.format(opt.dataset))
    from datetime import datetime
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

    model_path = opt.save_path

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    trainloader, testloader, valloader, (OriMax, OriMin), (MAX,MIN) = data_loader(opt)
    if not opt.normalize_location:
        OriMin[1] = 0
        OriMin[2] = 0
        OriMax[1] = 1
        OriMax[2] = 1

    T = MAX[0]
    if opt.num_types == 1:
        S = [[MIN[2],MAX[2]],[MIN[3],MAX[3]]]
    else:
        S = [[MIN[3],MAX[3]],[MIN[4],MAX[4]]]
    print('Time & Space Range', T, S)

    
    if opt.num_types > 1:
        loc_dim = 3
    else:
        loc_dim = 2



    if opt.model == 'SMASH_GMM':
        score_model = smash_intensity_score(opt.num_units, opt.cond_dim, num_types=opt.num_types, T=T, S=S, K_trig=opt.K_trig, kernel_type='GMM').to(device)
        # score_model = SMASH_GMM(opt.num_units, opt.cond_dim, num_types=opt.num_types, T=T, S=S, K_trig=opt.K_trig).to(device)
    elif opt.model == 'SMASH_decouple':
        # score_model = SMASH_decouple(opt.num_units, opt.cond_dim, num_types=opt.num_types, T=T, S=S).to(device)
        score_model = smash_intensity_score(opt.num_units, opt.cond_dim, num_types=opt.num_types, T=T, S=S, K_trig=opt.K_trig, kernel_type='decouple').to(device)
    elif opt.model == 'SMASH':
        # score_model = SMASH(opt.num_units, opt.cond_dim, num_types=opt.num_types, T=T, S=S).to(device)
        score_model = smash_intensity_score(opt.num_units, opt.cond_dim, num_types=opt.num_types, T=T, S=S, K_trig=opt.K_trig, kernel_type='vanilla').to(device)
    transformer = Transformer_ST(
            d_model=opt.cond_dim,
            d_rnn=opt.cond_dim*4,
            d_inner=opt.cond_dim*2,
            n_layers=opt.n_layers,
            n_head=opt.n_head,
            d_k=opt.d_k,
            d_v=opt.d_v,
            dropout=0.0,
            device=device,
            loc_dim = loc_dim,
            CosSin = True,
            num_types=1
        ).to(device)

    Model = SMSTPP(score_model, T, S, opt.estimator, transformer, device, 
    alpha_s=opt.alpha_s, alpha_CE=opt.alpha_CE, grid_t=opt.grid_t, grid_s=opt.grid_s, num_types=opt.num_types,
    num_noise=opt.num_noise, sigma_t=opt.sigma_t, sigma_s=opt.sigma_s, noise_type=opt.noise_type, with_survival=opt.with_survival, spatial_weight_looser_factor=opt.spatial_weight_looser_factor,
    eval_grid_t=opt.eval_grid_t, eval_grid_s=opt.eval_grid_s, intensity_grid_t=opt.intensity_grid_t, intensity_grid_s=opt.intensity_grid_s, identity_weight=opt.identity_weight).to(device)

    if opt.mode == 'test' or opt.mode == 'debug':
        total_epochs = 0
        Model.load_state_dict(torch.load(model_path+f"model_{opt.dataset}_{opt.model}_{'mle'}_{str(opt.seed)}_with_survival_{opt.with_survival}.pkl"))
        print('Weight loaded!!')
    total_params = sum(p.numel() for p in Model.parameters())
    print(f"Number of parameters: {total_params}")

    # training
    optimizer = AdamW(Model.parameters(), lr = opt.lr, betas = (0.9, 0.99))
    

    min_loss_test = 1e20
    best_ll = -1e20
    best_sll = -1e20
    best_tll = -1e20
    best_label_ll = -1e20
    best_label_acc = -1e20


    gt_model = choose_kernel(opt.dataset, S)
    # gt_model = None


    if opt.mode == 'train' or opt.mode == 'debug':
        loss_lst = []
        for itr in tqdm(range(opt.total_epochs)):
            Model.train()

            print('epoch:{}'.format(itr))
            loss_all, total_num = 0.0, 0.0
            # loss_lst = []
            for batch in trainloader:
                # if itr == 1:
                #     print(batch[0])
                loss, bs = Model(batch, 'train')
                # loss = loss.sum() / bs
                loss_lst.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                loss_all += loss.item() * bs
                torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
                optimizer.step() 
                # step += 1
                total_num += bs

            with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
                torch.cuda.empty_cache()
            print('------- Training ---- Epoch: {} ;  Loss: {} --------'.format(itr, loss_all/total_num/(T)))

            if itr % 10 == 0:
                Model.eval()
                total_ll = 0.0
                total_tll = 0.0
                total_sll = 0.0
                total_num = 0.0
                if opt.num_types > 1:
                    total_label_count = 0.0
                    total_label_ll = 0.0
                else:
                    total_label_ll = None
                for batch in testloader:
                    if opt.num_types == 1:
                        t_ll, s_ll, bs = Model(batch, "test")
                    else:
                        t_ll, s_ll, label_ll, bs, label_acc = Model(batch, "test")
                    ll = t_ll + s_ll
                    if opt.num_types > 1:
                        ll += label_ll
                        total_label_ll += label_ll.sum().item()
                        total_label_count += label_acc.item() * bs
                    total_tll += t_ll.sum().item()
                    total_sll += s_ll.sum().item()
                    total_ll += ll.sum().item()
                    non_pad_mask = get_non_pad_mask(batch[0])
                    total_num += non_pad_mask.sum().item()

                print('------- Testing ---- Epoch: {} ;  TLL: {} ; SLL: {} ; LL: {} ; Label LL: {} ; Label Acc: {} --------'.format(itr, total_tll/total_num, total_sll/total_num, total_ll/total_num\
                    , total_label_ll/total_num if opt.num_types > 1 else None, total_label_count/total_num if opt.num_types > 1 else None))
                if total_ll/total_num > best_ll:
                    best_ll = total_ll/total_num
                    best_sll = total_sll/total_num
                    best_tll = total_tll/total_num
                    if opt.num_types > 1:
                        best_label_ll = total_label_ll/total_num
                        best_label_acc = total_label_count/total_num
                    torch.save(Model.state_dict(), model_path+f"model_{opt.dataset}_{opt.model}_{opt.estimator}_{str(opt.seed)}_with_survival_{opt.with_survival}.pkl")
                    print('Model Saved to {}'.format(model_path+f"model_{opt.dataset}_{opt.model}_{opt.estimator}_{str(opt.seed)}_with_survival_{opt.with_survival}.pkl"))

                    # save the random state, for reproducibility
                    cpu_state = torch.get_rng_state()
                    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    np_state = np.random.get_state()
                    py_state = random.getstate()
                    test_lamb_temporal(testloader, Model, device, opt.estimator+'_'+opt.dataset+'_'+opt.model+'_'+str(opt.seed), gt_model)
                    if opt.dataset in ['Earthquake', 'COVID19', 'Citibike']:
                        plot_map_temperature(testloader, Model, device, opt.dataset+'_'+opt.model+'_'+opt.estimator+'_'+str(opt.seed), OriMax, OriMin)
                    torch.set_rng_state(cpu_state)
                    if cuda_state is not None:
                        torch.cuda.set_rng_state(cuda_state)
                    np.random.set_state(np_state)
                    random.setstate(py_state)

        print("Best LL", best_ll)
        print("Best SLL", best_sll)
        print("Best TLL", best_tll)
        print("Best Label LL", best_label_ll)
        print("Best Label Acc", best_label_acc)

        # plot the loss curve, save as pdf in the log path
        plt.figure()
        plt.plot(loss_lst, label='Loss')
        plt.legend()
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
        plt.savefig(model_path+f"loss_curve_{opt.dataset}_{opt.model}_{opt.estimator}_{str(opt.seed)}_with_survival_{opt.with_survival}.pdf")
        plt.close()

        if opt.mode == "train":
            #load best model
            Model.load_state_dict(torch.load(model_path+f"model_{opt.dataset}_{opt.model}_{opt.estimator}_{opt.seed}_with_survival_{opt.with_survival}.pkl"))
            # print("stop here")
            test_lamb_spatial(testloader, [Model], device, [opt.estimator],opt.dataset+'_'+opt.model+'_'+str(opt.seed), gt_model, S, T)
            if opt.dataset in ['Earthquake', 'COVID19', 'Citibike']:
                plot_map_temperature(testloader, Model, device, opt.dataset+'_'+opt.model+'_'+opt.estimator+'_'+str(opt.seed), OriMax, OriMin, n_traj=10)




    if opt.mode == "test" or opt.mode == "debug":
        Model.eval()
        # for debugging
        if opt.dataset in ['Earthquake', 'COVID19', 'Citibike']:
            plot_map_temperature(testloader, Model, device, opt.dataset+'_'+opt.model+'_'+opt.estimator+'_'+str(opt.seed), OriMax, OriMin, n_traj=10)
        test_lamb_temporal(testloader, Model, device, opt.estimator+'_'+opt.dataset+'_'+opt.model+'_'+str(opt.seed), gt_model)
        test_lamb_spatial(testloader, [Model], device, [opt.estimator],opt.dataset+'_'+opt.model+'_'+str(opt.seed), gt_model, S, T)
        

        for batch in testloader:

            Model.visualize(batch, opt.dataset, opt.seed)
    



