import src.sdes as sdes
import src.utils as utils
import torch
import torch.optim.swa_utils as swa_utils
import sys
import fire
import data
from src.discriminative_metrics import discriminative_metrics
from src.predictive_metrics import predictive_metrics
from src.check_independence import check_independence
import numpy as np


def main(
         noise_size=5,
         mlp_size=128,           # How big the layers in the various MLPs are.
         num_layers=2,
         batch_size=2048, 
         num_plot_samples=50,
         plot_locs=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
         propagations = 10,
         initial_adjust=0,
):
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    
    
    test_ts, data_size, test_dataloader = data.FitzHughNagumo_data(batch_size=batch_size, 
                                                                   device=device, phase='test')
    
    generator = sdes.DirectedChainSDE(noise_size, data_size, mlp_size, num_layers, initial_adjust).to(device)
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_generator.load_state_dict(torch.load("fitzhugh_generator.net"))
    generator.load_state_dict(averaged_generator.module.state_dict())
    
    
    # notice that since we normalize our data w.r.t. initial distribution, 
    # the initial is standard normal
    def initial_generator(size, device):
        return torch.randn(size, 3, device=device)
    
    utils.plot(test_ts, 
               generator, 
               test_dataloader, 
               num_plot_samples, 
               plot_locs, 
               initial_generator,
               propagations=propagations,
               addr=sys.path[0]+"/fig/FitzHughNagumo",
               channel=1)
    
    utils.plot2dhist(test_ts, 
                     generator, 
                     test_dataloader, 
                     plot_locs, 
                     initial_generator,
                     propagations=propagations,
                     addr=sys.path[0]+"/fig/FitzHughNagumo",
                     channel1=0,
                     channel2=1)
    sig_mmds = []
    for i in range(10):
        initial = initial_generator(batch_size, device)
        sig_mmd = utils.evaluate_sig_loss(test_ts, batch_size, test_dataloader, generator, initial, sig_depth=6, propagations=propagations)
        sig_mmds.append(sig_mmd)
    print(f"Sig-MMD is {np.mean(sig_mmds):4f}, {np.std(sig_mmds):4f}.")
    
    real_data = torch.cat([dat[0] for dat in test_dataloader], dim=0)
    prev_data = real_data.clone()
    
    with torch.no_grad():
        cor_maxs = []
        for i in range(10):
            for i in range(propagations):
                generated_data = []
                initial = initial_generator(real_data.size(0), device)
                generated_sample = generator(test_ts, prev_data.size(0), prev_data, initial, device)
                generated_data = generated_sample[:, :, 1:]
                cor_max = check_independence(real_data[..., 1:], generated_data)
            cor_maxs.append(cor_max)
        print(" Check Independence: ", np.mean(cor_maxs), np.std(cor_maxs))
    
    
    accs = []
    l1_losses = []
    real_data = real_data[..., 1:]
    for _ in range(10):
        accs.append(discriminative_metrics(real_data, generated_data, data_size, device, print_info=False))
        l1_losses.append(predictive_metrics(real_data, generated_data, data_size, device, print_info=False))
        print(f"Discriminative--{accs[-1]}, Predictive--{l1_losses[-1]}")
    print(f"Discriminative metrics: {np.mean(accs)}({np.std(accs)})")
    print(f"Predictive metrics: {np.mean(l1_losses)}({np.std(l1_losses)})")
    
    
if __name__ == '__main__':
    fire.Fire(main)