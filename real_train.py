import src.sdes as sdes
import src.utils as utils
import tqdm
import torch
import torchcde
import data
import torch.optim.swa_utils as swa_utils
import argparse
import signatory


model_sde = {
        "directed_chain": sdes.DirectedChainSDE,
        "classic": sdes.ClassicSDE,
        }
data_loading = {
    "stock": data.stock_data_loading,
    "equity": data.equity_data_loading,
    "electric": data.electric_data_loading}


class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super(DiscriminatorFunc, self).__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size
        self._module = sdes.MLP(1+hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=False)
        

    def forward(self, t, h):
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super(Discriminator, self).__init__()

        self._initial = sdes.MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.
        
        
        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=4/24,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()
    
    


def main(
        args,
        noise_size=10,
        hidden_size=16,
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        mlp_size=128,           # How big the layers in the various MLPs are.
        num_layers=2,          # How many hidden layers to have in the various MLPs.
        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr= 0.0001, #.0001, for stock      # Learning rate often needs careful tuning to the problem.
        discriminator_lr= 0.0001, # .0001, for stock
        batch_size=128,        # Batch size.
        steps=4000,            # How many steps to train both generator and discriminator for.
        weight_decay=0.00001,      # Weight decay.
        #sig_depth = 5,
        # Evaluation and plotting hyperparameters
        steps_per_print=50,                   # How often to print the loss.
):

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    
    sig_depth = args.sig_depth
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    else:
        print("Congratulations! CUDA is available!")
    
    ts, data_size, dataloader = data_loading[args.data_name](args.seq_len, batch_size, device)
    infinite_dataloader = (elem for it in iter(lambda: dataloader, None) for elem in it)
    print(args.data_name + ' dataset is ready, dim ' + str(data_size))
    
    generator = model_sde[args.model_type](noise_size, data_size, mlp_size, num_layers, time_included=False).to(device)
    discriminator = Discriminator(data_size, hidden_size, mlp_size, num_layers).to(device)
    averaged_generator = swa_utils.AveragedModel(generator)
    
    generator_optimiser = torch.optim.Adam(generator.parameters(), lr=generator_lr, 
                                          weight_decay=weight_decay)
    discriminator_optimiser = torch.optim.Adam(discriminator.parameters(), lr=discriminator_lr,
                                              weight_decay=weight_decay)
    
    trange = tqdm.tqdm(range(steps))
    swa_step_start = steps-100
    for step in trange:
        if (step+1)%2000 == 0:
            #generator_lr /= 10
            #discriminator_lr /= 10
            #weight_decay /= 10
            #generator_optimiser = torch.optim.SGD(generator.parameters(), lr=generator_lr, 
            #                                       weight_decay=weight_decay)
            #discriminator_optimiser = torch.optim.SGD(discriminator.parameters(), lr=discriminator_lr,
            #                                           weight_decay=weight_decay)
            generator_optimiser.param_groups[0]['lr'] /= 10
            discriminator_optimiser.param_groups[0]['lr'] /= 10
        real_samples,  = next(infinite_dataloader)
        re_sample_idx = torch.randint(high=real_samples.size(0), size=(real_samples.size(0),))
        initial = real_samples[:, 0][re_sample_idx]
        #initial = real_samples[:, 0]
        generated_samples = generator(ts, real_samples.size(0), real_samples, initial, device)
        generated_sig = signatory.signature(generated_samples, sig_depth, basepoint=True)
        generated_score = discriminator(generated_samples)
        
        real_samples = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(real_samples.size(0), 
                                                                       real_samples.size(1), 
                                                                       1), real_samples], dim=2)
        real_sig = signatory.signature(real_samples, sig_depth, basepoint=True)
        real_score = discriminator(real_samples)
        
        #loss = -torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
        loss = real_score-generated_score - torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
        
        loss.backward()
        ###
        #write the gradient update code
        for param in generator.parameters():
            param.grad *= -1
        ###

        generator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.step()
        discriminator_optimiser.zero_grad()
        
        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)
        
        if step > swa_step_start:
            averaged_generator.update_parameters(generator)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_unaveraged_loss = utils.evaluate_sig_loss(ts, batch_size, dataloader, generator, initial, 
                                                            sig_depth=sig_depth, time_included=False, special_initial=True)
            if step > swa_step_start:
                total_averaged_loss = utils.evaluate_sig_loss(ts, batch_size, dataloader, averaged_generator.module, initial, 
                                                              sig_depth=sig_depth, time_included=False, special_initial=True)
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
        
    torch.save(averaged_generator.state_dict(), args.data_name+'.net')
    


if __name__ == '__main__':
    #fire.Fire(main)
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data_name',
            choices=['stock','equity', 'electric'],
            default='stock',
            type=str)
    parser.add_argument(
            '--seq_len',
            help='sequence length',
            default=24,
            type=int)
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
    main(args)
