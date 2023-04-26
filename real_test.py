import src.sdes as sdes
import torch
import signatory
import data
import torch.optim.swa_utils as swa_utils
import argparse
from src.discriminative_metrics import discriminative_metrics
from src.predictive_metrics import predictive_metrics
from src.check_independence import check_independence
import matplotlib.pyplot as plt
import numpy as np



model_sde = {
        "directed_chain": sdes.DirectedChainSDE,
        "classic": sdes.ClassicSDE,
        }

data_loading = {
    "stock": data.stock_data_loading,
    "equity": data.equity_data_loading,
    "electric": data.electric_data_loading}

def main(
        args,
        model_address,
        batch_size=128,
        propagations = 10,
        noise_size=10,
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        mlp_size=128,           # How big the layers in the various MLPs are.
        num_layers=2, 
        plot_locs=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
         ):
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    #device = 'cpu'
    ts, data_size, dataloader = data_loading[args.data_name](args.seq_len, batch_size, device)
    real_data = torch.cat([dat[0] for dat in dataloader], dim=0)
    
    
    generator = model_sde[args.model_type](noise_size, data_size, mlp_size, num_layers, time_included=False).to(device)
    #generator = sdes.ClassicSDE(noise_size, data_size, mlp_size, num_layers, time_included=False).to(device)
    averaged_generator = swa_utils.AveragedModel(generator)
    
    averaged_generator.load_state_dict(torch.load(model_address))
    generator.load_state_dict(averaged_generator.module.state_dict())
    
    sig_depth = args.sig_depth
    prev_data = real_data.clone()
    real_sig = signatory.signature(real_data, sig_depth, basepoint=True)
    
    
    # eval timeGAN performance
    #tgan_data = torch.tensor(np.load("TGAN_fake_stock.npy"), dtype=torch.float32, device=device)
    #tgan_sig = signatory.signature(tgan_data, sig_depth, basepoint=True)
    #sig_mmd = torch.sum(torch.mean(tgan_sig-real_sig, dim=0)**2)**0.5
    #cor_max = check_independence(real_data, tgan_data)
    #print("TimeGAN sig: ", sig_mmd, " -- TimeGAN Indep. Score: ", cor_max)
    
    with torch.no_grad():
        if args.model_type != "directed_chain":
            propagations = 1
        cor_maxs = [] 
        sig_mmds = []
        for _ in range(10):
            for i in range(propagations):
                generated_data = []
                _sample_idx = torch.randint(high=prev_data.size(0), size=(prev_data.size(0),))
                initial = prev_data[:, 0][_sample_idx]
                #initial = prev_data[:, 0]
                generated_sample = generator(ts, prev_data.size(0), prev_data, initial, device)
                generated_data = generated_sample[:, :, 1:]
                cor_max = check_independence(real_data, generated_data)
            cor_maxs.append(cor_max)
            
            generated_sig = signatory.signature(generated_data, sig_depth, basepoint=True)
            
            sig_mmd = torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
            sig_mmds.append(sig_mmd.item())
        print(" Check Independence: ", np.mean(cor_maxs), np.std(cor_maxs))
        print(f"Sig-MMD is {np.mean(sig_mmds):4f}, {np.std(sig_mmds):4f}.")
            
        
    # evaluate sig-mmd
    
    print(f"Real Data Size: {real_data.shape}, Generated Data Size: {generated_data.shape}")
    accs = []
    l1_losses = []
    
    for _ in range(10):
        if args.data_name == 'electric':
            epochs = 2000
        else:
            epochs = 5000
        accs.append(discriminative_metrics(real_data, generated_data, data_size, device, print_info=False, epochs=epochs))
        if data_size > 1:
            l1_losses.append(predictive_metrics(real_data, generated_data, data_size, device, print_info=False))
            print(f"Discriminative--{accs[-1]}, Predictive--{l1_losses[-1]}")
        else:
            print(f"Discriminative--{accs[-1]}")
    print(f"Discriminative metrics: {np.mean(accs)}({np.std(accs)})")
    if data_size > 1:
        print(f"Predictive metrics: {np.mean(l1_losses)}({np.std(l1_losses)})")
    
    
    #colors = ['red', 'blue', 'yellow', 'black', 'green']
    #colors = ['blue']*50
    num_paths = 50
    for ch in range(data_size):
        real_first = True
        generated_first = True
        fig = plt.figure()
        i = 0
        for real_sample_ in real_data[:num_paths]:
            kwargs = {'label': 'Real'} if real_first else {}
            plt.plot(ts.cpu(), real_sample_[:, ch].cpu(), color='blue', linewidth=0.5, alpha=0.7, **kwargs)
            real_first = False
            i += 1
        i = 0
        for generated_sample_ in generated_data[:num_paths]:
            kwargs = {'label': 'Generated'} if generated_first else {}
            plt.plot(ts.cpu(), generated_sample_[:, ch].cpu(), color='red', ls="--", linewidth=0.5, alpha=0.7, **kwargs)
            generated_first = False
            i += 1
        plt.legend()
        plt.title(f"50 Samples from both real & generated Dim {ch}.")
        plt.tight_layout()
        fig.savefig('fig/' + args.data_name + "/samplePath5_dim_" + str(ch) + ".pdf")
    
    return


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data_name',
            choices=['stock','energy','electric'],
            default='stock',
            type=str)
    parser.add_argument(
            '--seq_len',
            help='sequence length',
            default=24,
            type=int)
    parser.add_argument(
            '--model_address',
            choices=['stock.net','equity.net', 'electric.net'],
            default='stock.net',
            type=str)
    parser.add_argument(
            '--sig_depth',
            help='signature_depth',
            default=4,
            type=int)
    parser.add_argument(
            '--model_type',
            choices=['directed_chain', 'classic'],
            default='directed_chain',
            type=str)
    args = parser.parse_args()
    main(args, args.model_address)
