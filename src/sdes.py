import torch
#import signatory


class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            model.append(torch.nn.Tanh())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


###################
# Now we define the MKSDEs.
#
# We begin by defining the generator SDE.
###################
class SDE(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super(SDE, self).__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)


    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)
    

class ClassicSDE(torch.nn.Module):
    
    def __init__(self, noise_size, data_size, mlp_size, num_layers, time_included=True):
        super(ClassicSDE, self).__init__()
        
        self.noise_size = noise_size
        self.data_size = data_size
        self.hidden_size = 2*data_size
        self._read_in = MLP(data_size, self.hidden_size, mlp_size, num_layers, tanh=False)
        self._read_out = torch.nn.Linear(self.hidden_size, data_size)
        self.drift = MLP(1+self.hidden_size, self.hidden_size, mlp_size, num_layers, tanh=False)
        self.diffusion = MLP(1+self.hidden_size, self.hidden_size*noise_size, mlp_size, num_layers, tanh=False)
        self.time_included = time_included
        
    def forward(self, ts, batch_size, nei, initial, device):
        # we hope ts is included in nei as an time dimension to make code easier
        # ts: tensor, (len) or list?
        # The current stage only works for 1d process
        batch, lenth, channels = nei.size()
        
        if self.time_included:
            channels -= 1
        assert batch==batch_size, "neighbors and sdes don't match in sample size"
        xs = torch.zeros(batch, lenth, self.hidden_size, device=device)
        xs[:, 0] = self._read_in(initial)
        
        times = ts.unsqueeze(0).unsqueeze(-1).expand(batch, ts.size(0), 1)
        for i in range(len(ts)-1):
            
            diffu = self.diffusion(torch.cat([xs[:, i],times[:, i]], dim=1)).view(batch, self.hidden_size, self.noise_size)
            bm_incre = torch.randn(batch, self.noise_size, 1, device=device)*torch.sqrt(ts[i+1]-ts[i])
            xs[:, i+1] = xs[:, i] \
                    + self.drift(torch.cat([xs[:, i],times[:, i]], dim=1))*(ts[i+1]-ts[i])\
                    + torch.matmul(diffu, bm_incre).squeeze(dim=2)
        
        #ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch, ts.size(0), 1)
        return torch.cat([times, self._read_out(xs)], dim=2)
            

class DirectedChainSDE(torch.nn.Module):
    
    def __init__(self, noise_size, hidden_size, mlp_size, num_layers, label_size=0, initial_adjust = 0, time_included=True):
        super(DirectedChainSDE, self).__init__()
        
        self.noise_size = noise_size
        self.hidden_size = hidden_size
        self.drift = MLP(1+hidden_size*2+label_size, hidden_size, mlp_size, num_layers, tanh=False)
        self.diffusion = MLP(1+hidden_size*2+label_size, hidden_size*noise_size, mlp_size, num_layers, tanh=False)
        self.labelize = False
        self.time_included = time_included
        
        
    def forward(self, ts, batch_size, nei, initial, device):
        # we hope ts is included in nei as an time dimension to make code easier
        # nei: tensor, (batch, len, channels)
        # ts: tensor, (len) or list?
        # The current stage only works for 1d process
        batch, lenth, channels = nei.size()
        if not self.time_included:
            nei = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(batch, lenth, 1), nei], dim=2)
        else:
            channels -= 1
        assert batch==batch_size, "neighbors and sdes don't match in sample size"
        #assert self.hidden_size==channels, f"neighbors(size={channels}) and sdes(size={self.hidden_size}) don't match in dimension"
        xs = torch.zeros(batch, lenth, self.hidden_size, device=device)
        xs[:, 0] = initial

        
        if self.labelize == False:
            for i in range(len(ts)-1):
                diffu = self.diffusion(torch.cat([xs[:, i],nei[:, i]], dim=1)).view(batch, self.hidden_size, self.noise_size)
                bm_incre = torch.randn(batch, self.noise_size, 1, device=device)*torch.sqrt(ts[i+1]-ts[i])
                xs[:, i+1] = xs[:, i] \
                    + self.drift(torch.cat([xs[:, i],nei[:, i]], dim=1))*(ts[i+1]-ts[i])\
                    + torch.matmul(diffu, bm_incre).squeeze(dim=2)
        else:
            for i in range(len(ts)-1):
                input_ = torch.cat([xs[:, i], nei[:, i], self.label], dim=1)
                diffu = self.diffusion(input_).view(batch, 
                                                    self.hidden_size, 
                                                    self.noise_size)
                
                bm_incre = torch.randn(batch, self.noise_size, 1, device=device)*torch.sqrt(ts[i+1]-ts[i])
                xs[:, i+1] = xs[:, i] \
                    + self.drift(input_)*(ts[i+1]-ts[i])\
                    + torch.matmul(diffu, bm_incre).squeeze(dim=2)
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch, ts.size(0), 1)
        return torch.cat([ts, xs], dim=2)
    
    def loadLabel(self, label):
        self.label = label
        self.labelize = True

    
class OpinionSDE(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super(OpinionSDE, self).__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)


    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        b = x.size(0)
        
        diff = x.view(b, 1, self._hidden_size) - x.view(1, b, self._hidden_size)
        diff = diff.view(-1, self._hidden_size)
        t = t.expand(diff.size(0), 1)
        
        tx = torch.cat([t, diff], dim=1)
        drift = self._drift(tx).view(b, -1, self._hidden_size)
        diffusion = self._diffusion(tx).view(b, -1, self._hidden_size, self._noise_size)
        
        return torch.mean(drift, dim=1), torch.mean(diffusion, dim=1)