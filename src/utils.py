import torch
import matplotlib
import matplotlib.pyplot as plt
import signatory
import numpy as np
from scipy import interpolate

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


plt.rc('xtick', labelsize=22)   #22 # fontsize of the tick labels
plt.rc('ytick', labelsize=22)   #22 
plt.rc('legend', fontsize=18) 
plt.rc('axes', labelsize=25)
plt.rcParams["figure.figsize"] = (7.5, 6)






def plot(ts, generator, dataloader, num_plot_samples, plot_locs, initial_generator, propagations=1, addr="fig", channel=0):
    # Get samples
    print("=== Produce Plots ===")
    #real_samples, = next(iter(dataloader))
    real_samples = torch.cat([dat[0] for dat in dataloader], dim=0)
    assert num_plot_samples <= real_samples.size(0)
    
    corss = []

    with torch.no_grad():
        prev_samples = real_samples.clone()
        for i in range(propagations):
            initial = initial_generator(prev_samples.size(0), ts.device)
            generated_samples = generator(ts, prev_samples.size(0), prev_samples, initial, ts.device)
            prev_samples = generated_samples.clone()
            corss.append(time_independence1d(real_samples[..., 1+channel], generated_samples[..., 1+channel], plot_locs))
    corss = torch.tensor(corss)
    #===== plot correlation ===
    i = 0
    fig = plt.figure()
    for prop in plot_locs:
        time = int(prop * (real_samples.size(1) - 1))
        plt.plot(corss[:, i], label=f"time {time/real_samples.size(1):.1f}")
        plt.legend()
        plt.xlabel("Directed Chain Steps")
        plt.ylabel("Abs val")
        #plt.title("Abs corrs.")
        i += 1
    fig.savefig(addr + '/' + 'corr.pdf')
    
    generated_samples.to('cpu')
    generated_samples = generated_samples[..., 1+channel]
    real_samples = real_samples[..., 1+channel]
    
    # xlims used for opinion plots
    xlims = [[-2.2, 2.2],
              [-2.0, 2.0],
              [-1.8, 1.8],
              [-1.5, 1.5],
              [-0.8, 0.8],
              [-0.6, 0.6]]
    # xlims used for fitzhugh plots
    # xlims = [[-3.2, 3.5],
    #           [-3.5, 3.8],
    #           [-3.5, 4.0],
    #           [-4.0, 4.0],
    #           [-4.0, 4.0],
    #           [-4.1, 4.1]]
    # ylims for opinion plots
    ylims = [[0, 0.4],
              [0, 0.5],
              [0, 0.65],
              [0, 0.83],
              [0, 1.41],
              [0, 2.3]]
    
    # Plot histograms
    i = 0
    for prop in plot_locs:
        time = int(prop * (real_samples.size(1) - 1))
        real_samples_time = real_samples[:, time]
        generated_samples_time = generated_samples[:, time]
        fig = plt.figure()
        _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
                              density=True)
        bin_width = bins[1] - bins[0]
        num_bins = 1+int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)
        plt.hist(generated_samples_time.cpu().numpy(), bins=num_bins, alpha=0.7, label='Generated', color='crimson',
                 density=True)
        plt.legend()
        #plt.xlim(real_samples.min().item()-0.2, real_samples.max().item()+0.2)
        left, right = xlims[i]
        
        plt.xlim(left, right)
        plt.ylim(ylims[i])
        i += 1
        plt.xlabel('Value')
        plt.ylabel('Density')
        #plt.title(f'Marginal distribution at time {time/real_samples.size(1):.1f}.')
        plt.tight_layout()
        fig.savefig(addr + '/' + str(prop)+'.pdf')
        
        

    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]

    # Plot samples
    real_first = True
    generated_first = True
    fig = plt.figure()
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    for generated_sample_ in generated_samples:
        kwargs = {'label': 'Generated'} if generated_first else {}
        plt.plot(ts.cpu(), generated_sample_.cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        generated_first = False
    plt.legend()
    #plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    fig.savefig(addr + "/samplePath.pdf")
    
    real_first = True
    generated_first = True
    fig = plt.figure()
    i = 0
    colors = ['red', 'blue', 'yellow', 'black', 'green']
    for real_sample_ in real_samples[:5]:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color=colors[i], linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
        i += 1
    i = 0
    for generated_sample_ in generated_samples[:5]:
        kwargs = {'label': 'Generated'} if generated_first else {}
        plt.plot(ts.cpu(), generated_sample_.cpu(), color=colors[i], ls="--", linewidth=0.5, alpha=0.7, **kwargs)
        generated_first = False
        i += 1
    plt.legend()
    #plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    fig.savefig(addr + "/samplePath5.pdf")
    
    
    
def plot2dhist(ts, 
               generator, 
               dataloader, 
               plot_locs, 
               initial_generator, 
               propagations=1, 
               addr="fig", 
               channel1=0,
               channel2=1):
    # Get samples
    real_samples, = next(iter(dataloader))
    
    with torch.no_grad():
        prev_samples = real_samples.clone()
        for i in range(propagations):
            initial = initial_generator(prev_samples.size(0), ts.device)
            generated_samples = generator(ts, prev_samples.size(0), prev_samples, initial, ts.device)
            prev_samples = generated_samples.clone()
    
    generated_samples.to('cpu')
    generated_samples = generated_samples[..., (1+channel1, 1+channel2)]
    real_samples = real_samples[..., (1+channel1, 1+channel2)]

    # 
    i = 0
    vmaxs = [0.36, 0.16, 0.11, 0.18, 0.25, 0.29]
    # Plot histograms
    for prop in plot_locs:
        time = int(prop * (real_samples.size(1) - 1))
        real_samples_time = real_samples[:, time]
        generated_samples_time = generated_samples[:, time]
        real_h, x_edges, y_edges, _ = plt.hist2d(real_samples_time[:, 0].cpu().numpy(), 
                                                 real_samples_time[:, 1].cpu().numpy(),
                                                 bins=20,
                                                 density=True)
        
        fake_h, _, _, _ = plt.hist2d(generated_samples_time[:, 0].cpu().numpy(), 
                                     generated_samples_time[:, 1].cpu().numpy(), 
                                     bins=[x_edges, y_edges],
                                     density=True)
        
        fig = plt.figure()
        h = np.abs(real_h - fake_h)
        plt.pcolormesh(x_edges, y_edges, h.T, vmax=vmaxs[i])
        i += 1
        #plt.legend()
        plt.colorbar()
        plt.xlabel(f"Dim {1+channel1}")
        plt.ylabel(f"Dim {1+channel2}")
        #plt.title(f'Marginal distribution diff at time {time/real_samples.size(1):.1f}.')
        plt.tight_layout()
        fig.savefig(addr + '/' + str(prop)+'_hist2d.pdf')
        
    # real_h, x_edges, y_edges, _ = plt.hist2d(real_samples_time[:, 0].cpu().numpy(), 
    #                                          real_samples_time[:, 1].cpu().numpy(),
    #                                          bins=10,
    #                                          density=True)
        
    # xx, yy = np.meshgrid((x_edges[:-1]+x_edges[1:])/2, (y_edges[:-1]+y_edges[1:])/2)
    # tck = interpolate.bisplrep(xx, yy, np.transpose(real_h), s=8)
    # #xxnew, yynew = np.mgrid[xx[0, 0]:xx[-1, 0]:200j, yy[0, 0]:yy[0, -1]:200j]
    # smooth_h = interpolate.bisplev((x_edges[:-1]+x_edges[1:])/2, (y_edges[:-1]+y_edges[1:])/2, tck)
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(xx, yy, smooth_h, cmap='viridis')
    # #ax.plot_surface((y_edges[:-1]+y_edges[1:])/2, (x_edges[:-1]+x_edges[1:])/2, real_h, cmap='viridis')
    # plt.xlabel(r"$V_T$")
    # plt.ylabel(r"$\mathcal{w}_T$")
    # ax.set_zlim3d(0)
    # plt.tight_layout()
    # fig.savefig(addr+'/'+str(prop)+'_density_surface.pdf')
    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(xx, yy, real_h, cmap='viridis')
    # #ax.plot_surface((y_edges[:-1]+y_edges[1:])/2, (x_edges[:-1]+x_edges[1:])/2, real_h, cmap='viridis')
    # plt.xlabel(r"$V_T$")
    # plt.ylabel(r"$\mathcal{w}_T$")
    # ax.set_zlim3d(0)
    # ax.set_zticks([0,0.02,0.04,0.06,0.08,0.1, 0.12], 
    #               [str(0), str(0.08), str(0.16),str(0.24),str(0.32),str(0.40),str(0.48)])
    # plt.tight_layout()
    # fig.savefig(addr+'/'+str(prop)+'_density_surface_rough.pdf')
        
        
    


    
    
def time_independence1d(path1, path2, locs):
    # path1, path2 are of 1 channel
    cors = []
    for loc in locs:
        time = int(loc*(path1.size(1)-1))
        path1_sample_time = path1[:, time]
        path2_sample_time = path2[:, time]
        cov = torch.mean(path1_sample_time * path2_sample_time, dim=0) \
                - torch.mean(path1_sample_time, dim=0)\
                * torch.mean(path2_sample_time, dim=0)
        cor = cov/(path1_sample_time.std()*path2_sample_time.std())
        cors.append(torch.abs(cor).item())
    return cors
        
    
def evaluate_sig_loss(ts, batch_size, dataloader, generator, initial, sig_depth=4, time_included=True, special_initial=False, propagations=1):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            if special_initial:
                re_sample_idx = torch.randint(high=real_samples.size(0), size=(real_samples.size(0),))
                initial = real_samples[:, 0][re_sample_idx]
            prev_samples = real_samples.clone()
            for i in range(propagations):
                generated_samples = generator(ts, prev_samples.size(0), prev_samples, initial, ts.device)
                prev_samples = generated_samples.clone()
            # do need to specify basepoint?
            generated_sig = signatory.signature(generated_samples, sig_depth, basepoint=True)
            if not time_included:
                real_samples = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(real_samples.size(0), 
                                                                       real_samples.size(1), 
                                                                       1), real_samples], dim=2)
            real_sig = signatory.signature(real_samples, sig_depth, basepoint=True)
            #loss = torch.nn.functional.mse_loss(generated_sig, real_sig)
            
            loss = torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples


def evaluate_sig_loss2(ts, 
                       batch_size, 
                       dataloader, 
                       generator, 
                       sig_depth=5):
    device = ts.device
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        total_loss_mmd = 0
        for real_samples, real_labels in dataloader:
            real_samples = real_samples.to(device)
            real_labels = real_labels.to(device)
            generator.loadLabel(real_labels)
            re_sample_idx = torch.randint(high=real_samples.size(0), size=(real_samples.size(0),))
            initial = real_samples[:, 0][re_sample_idx]
            generated_samples = generator(ts, real_samples.size(0), real_samples, initial, ts.device)
            # do need to specify basepoint?
            generated_sig = signatory.signature(generated_samples, sig_depth, basepoint=False)
            
            real_samples = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(real_samples.size(0), 
                                                                       real_samples.size(1), 
                                                                       1), real_samples], dim=2)
            real_sig = signatory.signature(real_samples, sig_depth, basepoint=False)
            
            
            loss = torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
            loss_mmd = torch.sum(generated_sig-real_sig, dim=0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_loss_mmd += loss_mmd
    return total_loss / total_samples, (total_loss_mmd/total_samples).abs().mean() # same 1/10000 as NeuralSDE paper
'''
            total_loss += real_sig - generated_sig
            total_samples += batch_size
        total_loss /= total_samples
    return (total_loss.abs().mean()/10000).item()
'''



def discriminative_train_test_split(real_seq, generated_seq, ratio=0.8, batch_size=128):
    # real_seq: tensor
    # generated_seq: tensor
    
    real_seq = [[real_seq[i], 0] for i in range(real_seq.size(0))]
    generated_seq = [[generated_seq[i], 1] for i in range(generated_seq.size(0))]
    
    all_seq = real_seq + generated_seq
    length = [int(len(all_seq)*ratio), len(all_seq)-int(len(all_seq)*ratio)]
    train, test = torch.utils.data.random_split(all_seq, length)
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    return trainloader, testloader
    
def eval_acc(classifier, dataloader):
    tot_num = 0
    correct_pred = 0
    for X,y in dataloader:
        pred = classifier(X)
        correct_pred += sum((pred[:, 0] < pred[:, 1]) == y.to(pred.device)).item()
        tot_num += pred.size(0)
    return correct_pred / tot_num



def predictive_train_test_split(real_seq, generated_seq, batch_size=128):
    #real_seq = [[real_seq[i][:-1], real_seq[i][-1]] for i in range(real_seq.size(0))]
    #generated_seq = [[generated_seq[i][:-1], generated_seq[i][-1]] for i in range(generated_seq.size(0))]
    
    #all_seq = real_seq + generated_seq
    #length = [int(len(all_seq)*ratio), len(all_seq)-int(len(all_seq)*ratio)]
    train = torch.utils.data.TensorDataset(generated_seq[:, :-1, :-1], generated_seq[:, 1:, -1])
    test = torch.utils.data.TensorDataset(real_seq[:, :-1, :-1], real_seq[:, 1:, -1])
    trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)
    return trainloader, testloader

def eval_L1distance(predictor, dataloader):
    tot_loss = 0
    tot_num = 0
    loss_fn = torch.nn.L1Loss()
    for X,y in dataloader:
        pred = predictor(X)
        #tot_loss += torch.sum(torch.abs(y.to(pred.device) - pred)).item()
        tot_loss += loss_fn(y.to(pred.device), pred.squeeze())*pred.size(0)
        tot_num += pred.size(0)
    return tot_loss.item() / tot_num