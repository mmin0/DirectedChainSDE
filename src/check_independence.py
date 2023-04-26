import torch



def check_independence(real_data, generated_data, checking_locs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    # real_data, generated_data should have same length, and match as omega-by-omega
    # how to check?
    length = real_data.size(1)
    _max = 0
    for _loc in checking_locs:
        
        idx = min(int(_loc) * length, length-1)
        x1, x2 = real_data[:, idx], generated_data[:, idx]
        tensor_prod_mean = torch.mean(torch.matmul(x1.unsqueeze(-1), x2.unsqueeze(-2)), dim=0)
        mean_tensor_prod = torch.matmul(torch.mean(x1, dim=0).unsqueeze(-1), 
                                        torch.mean(x2, dim=0).unsqueeze(0))
        cov_mat = tensor_prod_mean - mean_tensor_prod        
        cor_mat = cov_mat / torch.matmul(torch.std(x1, dim=0).unsqueeze(-1), 
                                         torch.std(x2, dim=0).unsqueeze(0))
        _max = max(_max, torch.mean(torch.abs(cor_mat)).item())
    return _max
