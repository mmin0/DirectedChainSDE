import src.sdes as sdes
import src.utils as utils
import fire
import tqdm
import torch
import torch.optim.swa_utils as swa_utils
import signatory
import data




def main(
        noise_size=5,
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        mlp_size=128,           # How big the layers in the various MLPs are.
        num_layers=2,          # How many hidden layers to have in the various MLPs.
        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr=.05,      # Learning rate often needs careful tuning to the problem.
        batch_size=1024,        # Batch size.
        steps=2000,            # How many steps to train both generator and discriminator for.
        weight_decay=0.001,      # Weight decay.
        swa_step_start=1900,    # When to start using stochastic weight averaging.
        sig_depth = 6,
        # Evaluation and plotting hyperparameters
        steps_per_print=10                   # How often to print the loss.
):
    

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    else:
        print("Congratulations! CUDA is available!")
        
    ts, data_size, train_dataloader = data.FitzHughNagumo_data(batch_size=batch_size, device=device)
    
    #f = open(sys.path[0]+"/fig/log", "w")
    #sys.stdout = f
        
    # Data
    #ts, data_size, train_dataloader = get_data(batch_size=batch_size, device=device)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    generator = sdes.DirectedChainSDE(noise_size, data_size, mlp_size, num_layers).to(device)

    averaged_generator = swa_utils.AveragedModel(generator)

    
    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimiser = torch.optim.SGD(generator.parameters(), lr=generator_lr, 
                                          weight_decay=weight_decay)

    # Train both generator and discriminator.
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        if (step+1)%500 == 0:
            generator_optimiser.param_groups[0]['lr'] /= 10
        
        real_samples, = next(infinite_train_dataloader)
        # notice that since we normalize our data w.r.t. initial distribution, 
        # the initial is standard normal
        initial = torch.randn(real_samples.size(0), 3, device=device)
        generated_samples = generator(ts, real_samples.size(0), real_samples, initial, ts.device)
        generated_sig = signatory.signature(generated_samples, sig_depth, basepoint=True)
        real_sig = signatory.signature(real_samples, sig_depth, basepoint=True)
        loss = torch.sum(torch.mean(generated_sig-real_sig, dim=0)**2)**0.5
        
        loss.backward()

        generator_optimiser.step()
        generator_optimiser.zero_grad()

        
        if step > swa_step_start:
            averaged_generator.update_parameters(generator)

        if (step % steps_per_print) == 0 or step == steps - 1:
            total_unaveraged_loss = utils.evaluate_sig_loss(ts, batch_size, train_dataloader, generator, initial, sig_depth=sig_depth)
            if step > swa_step_start:
                total_averaged_loss = utils.evaluate_sig_loss(ts, batch_size, train_dataloader, averaged_generator.module, initial, sig_depth=sig_depth)
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
    
    torch.save(averaged_generator.state_dict(), 'fitzhugh_generator.net')
    


if __name__ == '__main__':
    fire.Fire(main)
    
