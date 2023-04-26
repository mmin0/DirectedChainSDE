import torch
from src.discriminative_metrics import discriminative_metrics
from src.predictive_metrics import predictive_metrics
from src.check_independence import check_independence
import data
import numpy as np
import signatory

data_loading = {
    "stock": data.stock_data_loading,
    "opinion": data.StochasticOpinion_data,
    "fitzhugh": data.FitzHughNagumo_data,
    "electric": data.electric_data_loading}



def main(data_name, model_name, sig_depth=4, seq_len=24, batch_size=1024):
    
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    #device = 'cpu'
    if data_name in ["stock", "electric"]:
        ts, data_size, dataloader = data_loading[data_name](seq_len, batch_size, device)
    else:
        ts, data_size, dataloader = data_loading[data_name](batch_size, device, phase='test')
    
    real_data = torch.cat([dat[0] for dat in dataloader], dim=0)
    
    generated_data = torch.load(f"{data_name}_fake_{model_name}.pt").to(device)
    #
    print(f"===Quality Test: {data_name}-{model_name}===")
    print(f"Real Data Size: {real_data.shape}, Generated Data Size: {generated_data.shape}")
    accs = []
    l1_losses = []

    real_sig = signatory.signature(real_data, sig_depth, basepoint=True)
    
    num_iters = 10
    sig_mmds = []
    
    if data_name in ['opinion', 'fitzhugh']:
        generated_data = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(generated_data.size(0), -1, 1),
                                    generated_data], dim=2)
    
    generated_sig = signatory.signature(generated_data, sig_depth, basepoint=True)
    sig_mmd = torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
    sig_mmds.append(sig_mmd.item())
    print(f"Sig-MMD is {np.mean(sig_mmds):4f}, {np.std(sig_mmds):4f}.")
        
        
    if data_name in ['opinion', 'fitzhugh']:
        real_data = real_data[..., 1:]
        generated_data = generated_data[..., 1:]
    if data_name == 'electric':
        epochs = 2000
    else:
        epochs = 5000
    
    cor_maxs = []
    for _ in range(num_iters):
        cor_maxs.append(check_independence(real_data, generated_data))
        #accs.append(discriminative_metrics(real_data, generated_data, data_size, device, print_info=False, epochs=epochs))
        #if data_size > 1:
        #    l1_losses.append(predictive_metrics(real_data, generated_data, data_size, device, print_info=False))
            
    print(f"Independence score: {np.mean(cor_maxs)}, {np.std(cor_maxs)}")
    #print(f"Discriminative metrics: {np.mean(accs)}({np.std(accs)})")
    #if data_size > 1:
    #    print(f"Predictive metrics: {np.mean(l1_losses)}({np.std(l1_losses)})")
    return


if __name__ == "__main__":
    import sys
    f = open("log.txt", "w")
    sys.stdout = f
    for data_name in ['stock', 'electric', 'opinion', 'fitzhugh']:
        for model_name in ['logsig', 'ctfp']:
            print("===== {}_{} =====".format(data_name, model_name))
            main(data_name, model_name)
    f.close()
            
            
            
            