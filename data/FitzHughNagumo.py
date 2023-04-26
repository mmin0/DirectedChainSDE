import torch
import pathlib
#import time
import matplotlib.pyplot as plt

# generate data
# FitzHugh-Nagumo is a standard model from neuroscience
# it has N neurons and P different neuron populations, and 
# the process is a 3-dim SDEs
# we only consider the case of only 1 group of neurons
# x = [V, w, y]

# we use parameters from: https://arxiv.org/pdf/1808.05530.pdf

a, b, c, ar, ad = 0.7, 0.8, 0.08, 1, 1
I, lam, Tmax, vT, V_rev, J = 0.5, 0.2, 1, 2, 1, 1
Gamma, Lam = 0.1, 0.5
sig_ext, sig_J = 0.5, 0.2

def S(v, lam=lam, Tmax=Tmax, vT=vT):
    return Tmax/(1+torch.exp(-lam*(v - vT)))


def Chi(y, Gamma=Gamma, Lam=Lam):
    ret = torch.zeros(y.size(), dtype=y.dtype, device=y.device)
    mask = torch.logical_and(y>0, y<1)
    ret = Gamma*torch.exp(-Lam/(1-(2*y-1)**2))
    ret[~mask] = 0
    return ret

def Sigma(v, y, ar=ar, ad=ad):    
    return torch.sqrt(ar*S(v)*(1-y) + ad*y)*Chi(y)

def f(t, x, I=I, a=a, b=b, c=c, ar=ar, ad=ad):
    f_ = torch.zeros(x.size(), device=x.device)
    f_[:, 0] = x[:, 0]-x[:, 0]**3/3 - x[:, 1] + I
    f_[:, 1] = c*(x[:, 0]+ a - b*x[:, 1])
    f_[:, 2] = ar*S(x[:, 0])*(1-x[:, 2]) - ad*x[:, 2]
    return f_

def g(t, x, sig_ext=sig_ext, ar=ar, ad=ad):
    g_ = torch.zeros(x.size(0), 3, 2, device=x.device)
    g_[:, 0, 0] = sig_ext
    g_[:, -1, -1] = Sigma(x[:, 0], x[:, 2])
    return g_


# xj is the whole data samples, xi is the j-th sample
def b(xi, xj, V_rev=V_rev, J=J):
    b_ = torch.zeros(xj.size(), device=xj.device)
    b_[:, 0] = J*(xi[0] - V_rev)*xj[:, 2]
    return torch.mean(b_, dim=0)

def beta(xi, xj, V_rev=V_rev, sig_J=sig_J):
    beta_ = torch.zeros(xj.size(), device=xi.device)
    beta_[:, 0] = -sig_J*(xi[0] - V_rev)*xj[:, 2]
    return torch.mean(beta_, dim=0)
    
    
def generate_data(dataset_size, device, phase, T=1):
    ts = torch.linspace(0, T, 1+T*100, device=device)
    xs = torch.zeros(dataset_size, 1+T*100, 3, device=device)
    xs[:, 0] = torch.randn(dataset_size, 3, device=device)* torch.sqrt(torch.tensor([0.4, 0.4, 0.05], device=device))\
                + torch.tensor([0, 0.5, 0.3], device=device)
    
    
    for i in range(T*100):
        f_ = f(ts[i], xs[:, i])
        g_ = g(ts[i], xs[:, i])
        dt = ts[i+1]-ts[i]
        root_dt = torch.sqrt(dt)
        xs[:, i+1] = xs[:, i] + f_*dt + torch.matmul(g_, torch.randn(dataset_size, 2, 1, device=device)).squeeze()*root_dt
        
        tensor_b = torch.zeros(dataset_size, 3, device=device)
        tensor_beta = torch.zeros(dataset_size, 3, device=device)
        for j in range(dataset_size):
            tensor_b[i] = b(xs[j, i], xs[:, i])
            tensor_beta[i] = beta(xs[j, i], xs[:, i])
        
        xs[:, i+1] += tensor_b*dt + tensor_beta*torch.randn(dataset_size, 1, device=device)*root_dt
    print(f"{phase} phase contains any NaN value: ", xs.isnan().any())
    torch.save(xs, "data/"+phase+"_FitzHughNagumo.pt")




def FitzHughNagumo_data(batch_size, device, phase="train", T=1):
    _here = pathlib.Path(__file__).resolve().parent
    
    ts = torch.linspace(0, T, T*100+1, device=device)
    if phase == "train":
        ys = torch.load(_here/"data"/"train_FitzHughNagumo.pt").to(device)
    else:
        assert phase=="test", "phase has to be either train or test"
        ys = torch.load(_here/"data"/"test_FitzHughNagumo.pt").to(device)
    ###################
    # To demonstrate how to handle irregular data, then here we additionally drop some of the data (by setting it to
    # NaN.)
    ###################
    dataset_size = ys.size(0)
    #ys_num = ys.numel()
    #to_drop = torch.randperm(ys_num)[:int(0.3 * ys_num)]
    #ys.view(-1)[to_drop] = float('nan')
    
    # normalize according to initial data
    y0 = ys[:, 0]
    ys = (ys - y0.mean(dim=0)) / y0.std(dim=0)
    
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, T*100+1, 1),
                    ys], dim=2)

    ###################
    # Package up.
    ###################
    data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    #ys_coeffs = torchcde.linear_interpolation_coeffs(ys)
    dataset = torch.utils.data.TensorDataset(ys)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return ts, data_size, dataloader
    


def plot(ts, samples, num_plot_samples):
    # Get samples
    real_samples = samples
    assert num_plot_samples <= real_samples.size(0)
    #real_samples = torchcde.LinearInterpolation(samples).evaluate(ts)
    real_samples = real_samples[:num_plot_samples]
    # Plot samples
    plot_locs = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    for prop in plot_locs:
        time_ = int(prop * (real_samples.size(1) - 1))
        real_samples_time = real_samples[:, time_]
        plt.figure()
        _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
                              density=True)
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Marginal distribution at time {time_}.')
        plt.tight_layout()
    
    real_first = True
    
    plt.figure()
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from generated distributions.")
    plt.tight_layout()
    plt.show()
    
    
    


if __name__=="__main__":
    T = 1
    cuda_id = 3
    
    """
    print("Job start.")
    start = time.time()
    is_cuda = torch.cuda.is_available()
    cuda_chip = "cuda:" + str(cuda_id)
    device = cuda_chip if is_cuda else 'cpu'
    
    dataset_size=8192
    
    generate_data(dataset_size=dataset_size, device=device, phase="train", T=T)
    generate_data(dataset_size=dataset_size, device=device, phase="test", T=T)
    end = time.time()
    print(f"Data has been generated. Total time usage is {(end-start)/60:.2f} minutes.")
    
    """
    
    
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    ts = torch.linspace(0, T, 1+T*100, device=device)
    ys = torch.load("data/train_FitzHughNagumo.pt", map_location=torch.device('cpu'))
    
    plot(ts, ys[..., 0], 2000)
    