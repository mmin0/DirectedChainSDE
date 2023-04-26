import torch
from .utils import discriminative_train_test_split, eval_acc

def discriminative_metrics(real_data, generated_data, channels, device, batch_size=128, print_info=False, epochs=5000):
    #discriminative_acc = 0
    trainloader, testloader = discriminative_train_test_split(real_data, generated_data, ratio=0.8, batch_size=batch_size)
    
    num_layers = 2
    
    class Classifier(torch.nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
            super(Classifier, self).__init__()
            self.rnn = torch.nn.LSTM(input_size=input_size, 
                                    hidden_size=hidden_size, 
                                    num_layers=num_layers, 
                                    batch_first=batch_first)
            self.linear = torch.nn.Linear(hidden_size*num_layers, 2)
            #self.nonlinear = torch.nn.Sigmoid()

        def forward(self, X):
            _, (h, _) = self.rnn(X)
            h = h.permute(1, 0, 2)
            #return self.nonlinear(self.linear(h.reshape(h.shape[0], -1)))
            return self.linear(h.reshape(h.shape[0], -1))
    
    classifier = Classifier(channels, max(channels//2, 1), num_layers).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    
    infinite_loader = (elem for it in iter(lambda: trainloader, None) for elem in it)
    for e in range(epochs):
        X, y = next(infinite_loader)
        optimizer.zero_grad()
        preds = classifier(X.to(device))
        loss = loss_fn(preds, y.to(device))
        loss.backward()
        optimizer.step()
        if print_info and (e+1)%50 == 0:
            acc = eval_acc(classifier, testloader)
            print(f"Epoch {e}, accuracy = {acc:.4f}")
    #return discriminative_acc
    return abs(0.5-eval_acc(classifier, testloader))
    

