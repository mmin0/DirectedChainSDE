import torch
import torchsde
import time
import torchcde
import matplotlib.pyplot as plt
import pathlib
#import argparse




class StochasticOpinion(torch.nn.Module):
    sde_type = 'ito'
    noise_type = 'scalar'
        
    def __init__(self, theta1, theta2, sigma):
        super().__init__()
        self.register_buffer('sigma', torch.as_tensor(sigma))
        self.register_buffer('theta1', torch.as_tensor(theta1))
        self.register_buffer('theta2', torch.as_tensor(theta2))
            
    def phi(self, r):
        mask = (r>=self.theta2+1)
        phi_ = self.theta1 * torch.exp(-0.01/(1-(r-self.theta2)**2)).view(r.size(0), 1)
        phi_[mask] = 0
        return phi_
            
            
    def f(self, t, y):
        tensor = torch.zeros(y.size(), device=y.device)
        for i in range(y.size(0)):
            temp = y[i]-y
            tensor[i] = torch.mean(self.phi(torch.sum(temp**2, dim=1)**0.5)*temp, dim=0)
        return -tensor # of shape (batch, channels)
        
    def g(self, t, y):
        return self.sigma.expand(y.size(0), 1, 1)
    
def generate_data(dataset_size, device, phase, t_size=2):
    so_sde = StochasticOpinion(theta1=6, theta2=0.2, sigma=0.1).to(device)
    y0 = torch.rand(dataset_size, device=device).unsqueeze(-1) * 4 - 2
    ts = torch.linspace(0, t_size - 1, (t_size-1)*100+1, device=device)
    ys = torchsde.sdeint(so_sde, y0, ts, dt=1e-2)
    torch.save(ys, "data/"+phase+"_StochasticOpinion.pt")
    
    
def plot(ts, samples, num_plot_samples):
    # Get samples
    real_samples = samples.permute(1, 0, 2)
    assert num_plot_samples <= real_samples.size(0)
    #real_samples = torchcde.LinearInterpolation(samples).evaluate(ts)
    real_samples = real_samples[:num_plot_samples]
    # Plot samples
    
    real_first = True
    
    fig = plt.figure()
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from generated distributions.")
    plt.tight_layout()
    fig.savefig("samplePath(StochasticOpinion).pdf")
    


def StochasticOpinion_data(batch_size, device, t_size=2, phase="train", initial_adjust=0):
    
    torch.manual_seed(0)
    _here = pathlib.Path(__file__).resolve().parent
    
    ts = torch.linspace(0, t_size - 1, (t_size-1)*100+1, device=device)
    if phase == "train":
        ys = torch.load(_here/"data"/"train_StochasticOpinion.pt").to(device) + initial_adjust
    else:
        assert phase=="test", "phase has to be either train or test"
        ys = torch.load(_here/"data"/"test_StochasticOpinion.pt").to(device) + initial_adjust
    ###################
    # To demonstrate how to handle irregular data, then here we additionally drop some of the data (by setting it to
    # NaN.)
    ###################
    dataset_size = ys.size(1)
    ys_num = ys.numel()
    to_drop = torch.randperm(ys_num)[:int(0.3 * ys_num)]
    ys.view(-1)[to_drop] = float('nan')
    
    # normalize according to initial data
    #y0_flat = ys[0].view(-1)
    #y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat))
    #ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()
    
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, (t_size-1)*100+1, 1),
                    ys.transpose(0, 1)], dim=2)

    ###################
    # Package up.
    ###################
    data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return ts, data_size, dataloader
    

if __name__ == "__main__":
    
    t_size=2
    
    print("Job start.")
    start = time.time()
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    
    dataset_size=8192
    
    generate_data(dataset_size=dataset_size, device=device, phase="train", t_size=t_size)
    generate_data(dataset_size=dataset_size, device=device, phase="test", t_size=t_size)
    end = time.time()
    print(f"Data has been generated. Total time usage is {(end-start)/60:.2f} minutes.")
    
    
    """
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    ts = torch.linspace(0, t_size-1, 1+(t_size-1)*100, device=device)
    ys = torch.load("data/test_StochasticOpinion.pt", map_location=torch.device('cpu'))
    plot(ts, ys, 100)
    """
    
    
        
        
