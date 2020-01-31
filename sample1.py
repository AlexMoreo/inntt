import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from inntt import InteractiveNeuralTrainer
from inntt import *
import time
import logging

""" The function we try to learn in this toy example """
INPUT_SIZE = 100
OUTPUT_SIZE = 10
unknown_transformation = torch.rand(INPUT_SIZE,OUTPUT_SIZE)
unknown_activation = F.tanh
def unknown_function(shape):
    X = Variable(torch.rand(shape))
    y = Variable(unknown_activation(torch.mm(X.data, unknown_transformation)))
    return X, y

""" A simple network to learn it """
class Net(nn.Module):
    def __init__(self, input_size, out_size, drop_prob=0.5):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, out_size)
        self.drop_prob = drop_prob

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.drop_prob, self.training)
        x = F.relu(self.fc2(x))
        return x

def main():
    net = Net(INPUT_SIZE, OUTPUT_SIZE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X, y = unknown_function((10000, INPUT_SIZE))
    Xval, yval = unknown_function((1000, INPUT_SIZE))

    # optional: define a custom logger to catch all interactions
    logger = logging.getLogger(net.__class__.__name__)
    fh = logging.FileHandler('log_hist.txt')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # instantiate the neural trainer
    innt = InteractiveNeuralTrainer(logger=logger)

    # adds two interactions to triggers 'w' and 's', to increase/decrease the learning rate of an optimizer; increase is
    # to be performed at a factor of x10 and decrease x 1/10
    innt.add_interaction('w', increase_lr(optimizer=optimizer, factor=10.))
    innt.add_interaction('s', decrease_lr(optimizer=optimizer, factor=.1))

    # same for the 'weight_decay'
    innt.add_interaction('a', decrease_weight_decay(optimizer=optimizer, factor=.5))
    innt.add_interaction('d', increase_weight_decay(optimizer=optimizer, factor=2.))

    # the four interactions above (and many others) can be summarized with the following shortcuts
    # innt.add_optim_param_adapt('ws', optimizer, 'lr', inc_factor=10.)
    # innt.add_optim_param_adapt('da', optimizer, 'weight_decay', inc_factor=2.)
    # e.g., the first one takes two triggers (w=increase/s=decrease) to modify the 'lr' of the 'optimizer' by an
    # increasing factor of x10  and a default decreasing factor of 1/inc_factor (similar for weight_decay)

    # adapts the dropout probability, this time by constant increments (instead of scaling with a factor), and fixing a
    # max and min values. The 'drop_prob' attribute should exists as net.drop_prob as it is accessed through getattr
    innt.add_interaction('p', adapt_net_attr(net, 'drop_prob', increment=0.1, lmax=1))
    innt.add_interaction('o', adapt_net_attr(net, 'drop_prob', increment=-0.1, lmin=0))

    # interaction to invoke the validation routine on the fly (!)
    innt.add_interaction('v', validation(net, Xval, yval, criterion))

    # add interactions to quick-save and quick-load the net. These can be established to be 'synchronized' in order to
    # prevent them to collide with an optimizer.step() modification or the like (which I don't know if are locked).
    # Synchronized functions have to be explicitly called (e.g., at the begining of each training loop) with innt.synchronize()
    innt.add_interaction('q', quick_save(net), synchronized=True)
    innt.add_interaction('e', quick_load(net), synchronized=True)

    # adds the 'reboot' function to reinit the network params; this is equivalent to restart from scratch whenever the
    # optimizer or the net reaches a unstable configuration (e.g., containing nan values)
    innt.add_interaction('r', reboot(net, optimizer, input_size=INPUT_SIZE, out_size=OUTPUT_SIZE), synchronized=True)

    # runs the daemon, which will listen to the action triggers and operate in the shadows
    innt.start()

    for i in range(10000):
        innt.synchronize() # call the synchronized events which are queued (the rest of actions are undertaken on the fly)
        optimizer.zero_grad()
        y_ = net.forward(X)
        loss = criterion(y_, y)
        loss.backward()
        optimizer.step()

        print('[step {}]: loss={:.8f}, lr={:.8f}, weight_decay={:.8f}, drop_prob={:.4f}'.format(
            i, loss.item(), inspect_param(optimizer, 'lr'), inspect_param(optimizer, 'weight_decay'), net.drop_prob)
        )
        time.sleep(.2) # <- to simplify the visualization


if __name__ == '__main__':
    main()