import torch
from .utils import predictive_train_test_split, eval_L1distance

def predictive_metrics(real_data, generated_data, channels, device, print_info=False, batch_size=128):
    #discriminative_acc = 0
    trainloader, testloader = predictive_train_test_split(real_data, generated_data, batch_size=batch_size)
    
    num_layers = 2
    
    class Predictor(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
            super(Predictor, self).__init__()
            self.rnn = torch.nn.LSTM(input_size=input_size, 
                                      hidden_size=hidden_size, 
                                      num_layers=num_layers, 
                                      batch_first=batch_first)
            self.linear = torch.nn.Linear(hidden_size, 1)
            self.nonlinear = torch.nn.Sigmoid()

        def forward(self, X):
            output, _ = self.rnn(X)
            return self.nonlinear(self.linear(output))
    
    predictor = Predictor(channels-1, channels//2, num_layers).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    loss_fn = torch.nn.L1Loss()
    
    epochs = 5000
    infinite_loader = (elem for it in iter(lambda: trainloader, None) for elem in it)
    for e in range(epochs):
        X, y = next(infinite_loader)
        optimizer.zero_grad()
        preds = predictor(X.to(device))
        loss = loss_fn(preds.squeeze(), y.to(device))
        loss.backward()
        optimizer.step()
        if print_info and (e+1)%50 == 0:
            distance = eval_L1distance(predictor, testloader)
            print(f"Epoch {e}, l1 distance = {distance:.4f}")
    #return discriminative_acc
    return eval_L1distance(predictor, testloader)
    

