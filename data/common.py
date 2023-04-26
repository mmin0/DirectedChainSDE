import torch


def dataloader(dataset, batch_size):
    generator = torch.Generator().manual_seed(321)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                       generator=generator)
