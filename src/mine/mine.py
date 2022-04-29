import torch
from torch import nn
from torch.utils.data import DataLoader
from math import log

class MINE(nn.Module):
    def __init__(self, T=None, ema_coeff:float = 0.1, neurons:int = 12):
        super(MINE, self).__init__()
        self.alpha = ema_coeff                                                      # weight coefficient for exponential moving average, higher alpha discounts older observations faster
        self.running_mean = None                                                # running exponential moving average

        # statistics network
        if T is None:
            self.T = nn.Sequential(
                nn.Linear(2, neurons),
                nn.ReLU(),
                nn.Linear(neurons, neurons),
                nn.ReLU(),
                nn.Linear(neurons, 1),
            )
        else:
            self.T = T

        if torch.cuda.is_available():
            # n_gpus = torch.cuda.device_count()
            # gpu_id = torch.cuda.current_device()
            # gpu_prop = torch.cuda.get_device_properties(gpu_id)
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.T.to(self.device)


    def forward(self, xy):

        t = self.T(xy).mean()

        # estimate marginal by shuffling along batch axis
        ymarg = xy[torch.randperm(xy.shape[0]),1]
        xymarg = torch.stack([xy[:,0],ymarg],1)
        t_marg = self.T(xymarg)

        running_mean = self.update_ema(torch.mean(torch.exp(t_marg)).detach())

        t_log = MINE_EMA_loss.apply(t_marg, running_mean)
        #t_log = torch.logsumexp(t_marg,0)-log(xy.shape[0])

        return -t + t_log                                                       # gradient ascent => -loss => -(t-t_log)

    def update_ema(self, new_mean):
        if self.running_mean is None:
            self.running_mean = new_mean
        else:
            self.running_mean = self.alpha*new_mean + (1-self.alpha)*self.running_mean

        return self.running_mean

    def train_loop(self, X, Y, max_iter:int = 100, batch_size:int = 500, opt=None):

        old_param = self.flatten_parameters()

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)

        mi = []
        for i in range(1, max_iter + 1):
            mi_ = 0
            for batch, xy in enumerate(DataLoader(torch.stack([X,Y],1), batch_size, shuffle=True)):
                opt.zero_grad()
                loss = self.forward(xy)
                mi_ += loss.item()
                loss.backward()
                opt.step()

            mi.append(-mi_/(batch+1))

            new_param = self.flatten_parameters()
            param_norm = torch.linalg.norm(new_param - old_param)
            old_param = new_param

            if i % (max_iter // 4) == 0:
                print(f"It {i} - MI: {mi[-1]}")

            # if param_norm.item() < 0.005:
            #     print(f"Converged in {i} iterations - MI: {mi[-1]}")
            #     break

        return mi

    def flatten_parameters(self):

        p = torch.tensor([],device=self.device)

        for param in self.parameters():
             p = torch.cat([p, param.data.flatten()])

        return p


class MINE_EMA_loss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, ema):

        ctx.save_for_backward(input)
        ctx.ema = ema

        t_lse_mean = input.exp().mean().log()

        return t_lse_mean

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors

        grad = grad_output*torch.exp(input).detach()/ctx.ema/input.shape[0]

        return grad, None
