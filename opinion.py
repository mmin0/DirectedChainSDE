import src.sdes as sdes
import src.utils as utils
import fire
import tqdm
import torch
import torch.optim.swa_utils as swa_utils
import signatory
import data



def main(
        noise_size=3,
        # Architectural hyperparameters. These are quite small for illustrative purposes.
        mlp_size=128,           # How big the layers in the various MLPs are.
        num_layers=2,          # How many hidden layers to have in the various MLPs.
        # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
        generator_lr=.001,      # Learning rate often needs careful tuning to the problem.
        batch_size=1024,        # Batch size.
        steps=2000,            # How many steps to train both generator and discriminator for.
        weight_decay=0.0,      # Weight decay.
        swa_step_start=1900,    # When to start using stochastic weight averaging.
        sig_depth = 8,
        initial_adjust = 0,
        # Evaluation and plotting hyperparameters
        steps_per_print=10,                   # How often to print the loss.
        num_plot_samples=50,                  # How many samples to use on the plots at the end.
        plot_locs=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),  # Plot some marginal distributions at this proportion of the way along.
):
    

    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    else:
        print("Congratulations! CUDA is available!")
        
    ts, data_size, train_dataloader = data.StochasticOpinion_data(batch_size=batch_size, device=device, initial_adjust=initial_adjust)
    
    #f = open(sys.path[0]+"/fig/log", "w")
    #sys.stdout = f
        
    # Data
    #ts, data_size, train_dataloader = get_data(batch_size=batch_size, device=device)
    print("Data is ready.")
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
    print("Infinite dataLoader is ready.")
    generator = sdes.DirectedChainSDE(noise_size, data_size, mlp_size, num_layers, 
                                      initial_adjust=initial_adjust).to(device)

    averaged_generator = swa_utils.AveragedModel(generator)    
    
    generator_optimiser = torch.optim.Adam(generator.parameters(), lr=generator_lr, 
                                          weight_decay=weight_decay)

    # Train both generator and discriminator.
    print("Model is ready, start training ...")
    trange = tqdm.tqdm(range(steps))
    for step in trange:
        if (step+1)%300 == 0:
            generator_optimiser.param_groups[0]['lr'] /= 10
        
        real_samples, = next(infinite_train_dataloader)
        initial = torch.rand(real_samples.size(0), device=device).unsqueeze(-1) * 4 - 2
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
    
    torch.save(averaged_generator.state_dict(), 'opinion_generator.net')
    


if __name__ == '__main__':
    fire.Fire(main)
