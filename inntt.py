#!/usr/bin/env python
import sys, os
import termios
import contextlib
import threading
from queue import Queue
import torch
import torch.nn as nn
from collections import OrderedDict

# TODO: conditional actions ?

class InteractiveNeuralTrainer(threading.Thread):

    def __init__(self, logger=None):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self._listeners = OrderedDict()
        self._sync_triggers = set()
        self._sync_queue = Queue()
        self.logger = logger

    def add_interaction(self, trigger, action, synchronized=False):
        assert isinstance(trigger, str) and len(trigger) == 1, 'a single char should be specified as the trigger'
        assert hasattr(action, '__call__'), 'the action should be callable'
        assert trigger not in self._listeners, '{} already in use'.format(trigger)

        self._listeners[trigger] = action
        if synchronized:
            self._sync_triggers.add(trigger)

        if trigger=='h':
            print('warning: {} will hidden the "help" interaction'.format(action.__name__))

    def add_optim_param_adapt(self, triggers, optimizer, param, inc_factor=1.1, dec_factor=None, synchronized=False):
        assert len(triggers)==2, 'two triggers requested to increase/decrease the parameter'
        assert inc_factor > 1, 'invalid inc_factor, must be greater than one'
        if dec_factor is None:
            dec_factor = 1./inc_factor
        self.add_interaction(triggers[0], adapt_optim_param(optimizer, param, inc_factor), synchronized=synchronized)
        self.add_interaction(triggers[1], adapt_optim_param(optimizer, param, dec_factor), synchronized=synchronized)

    def run(self):
        assert len(self._listeners) > 0, 'no interactions available'
        if 'h' not in self._listeners:
            self.add_interaction('h', self.help)
        with raw_mode(sys.stdin):
            print(self.__class__.__name__+' listening!')
            self.help()
            while True:
                trigger = sys.stdin.read(1)
                if trigger in self._listeners:
                    if trigger in self._sync_triggers:
                        self._sync_queue.put(trigger)
                    else:
                        if self.logger:
                            self.logger.info("{} -> {}".format(trigger, self._listeners[trigger].__name__))
                        self._listeners[trigger]()

    def synchronize(self):
        while not self._sync_queue.empty():
            trigger = self._sync_queue.get()
            print('\tsync: {}({})'.format(trigger, self._sync_queue.qsize()))
            self._listeners[trigger]()

    def help(self):
        print('\tTrigger\t->\taction')
        for trigger in self._listeners.keys():
            print('\t{}\t->\t{}'.format(trigger, self._listeners[trigger].__name__))


# I think this only works under Linux's terminal
@contextlib.contextmanager
def raw_mode(file):
    #from https://stackoverflow.com/questions/11918999/key-listeners-in-python
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


def inspect_param(optimizer, param):
    for param_group in optimizer.state_dict()['param_groups']:
        return param_group[param]

def __adapt_value(oldval, factor, increment, lmin, lmax, epsilon=1e-4):
    if oldval == 0 and factor>1 and increment==0: # the value is dead
        return epsilon

    newval = oldval * factor + increment
    if lmin is not None:
        newval = max(newval, lmin)
    if lmax is not None:
        newval = min(newval, lmax)
    return newval

def adapt_optim_param(optimizer, param, factor=1, increment=0, lmax=None, lmin=None):
    #assert param in optimizer.state_dict(), 'unknown parameter {} for the optimizer'.format(param)
    assert factor!=0, 'factor should be != 0'
    assert not (factor == 1 and increment == 0), 'setting factor=1 and increment=0 will have no effect'
    def adapt_optimizer_param_():
        state_dict = optimizer.state_dict()
        for param_group in state_dict['param_groups']:
            param_group[param] = __adapt_value(param_group[param], factor, increment, lmin, lmax)
            print('\t%s updated: %.8f' % (param, param_group[param]), end='\n')
        optimizer.load_state_dict(state_dict)
    return adapt_optimizer_param_

def adapt_net_attr(net, attr, factor=1, increment=0, lmin=None, lmax=None):
    assert isinstance(attr, str), 'attr should be a str'
    assert hasattr(net, attr), '{} does not have attribute {}'.format(net.__class__.__name__, attr)
    assert factor != 0, 'factor should be != 0'
    assert not (factor == 1 and increment==0), 'setting factor=1 and increment=0 will have no effect'
    def adapt_net_attr_():
        setattr(net, attr, __adapt_value(getattr(net, attr), factor, increment, lmin, lmax))
    return adapt_net_attr_

def increase_lr(optimizer, factor=1.1):
    assert factor>1, 'the factor should be >1'
    return adapt_optim_param(optimizer, 'lr', factor)

def decrease_lr(optimizer, factor=0.9):
    assert factor<1, 'the factor should be <1'
    return adapt_optim_param(optimizer, 'lr', factor)

def increase_weight_decay(optimizer, factor=1.1):
    assert factor>1, 'the factor should be >1'
    return adapt_optim_param(optimizer, 'weight_decay', factor)

def decrease_weight_decay(optimizer, factor=0.9):
    assert factor<1, 'the factor should be <1'
    return adapt_optim_param(optimizer, 'weight_decay', factor)

def quick_load(net, save_dir='checkpoint'):
    saved_path = os.path.join(save_dir, net.__class__.__name__ + '_quick_save')
    def quick_load_():
        print('\tloading ' + saved_path)
        net.load_state_dict(torch.load(saved_path).state_dict())
    return quick_load_

def quick_save(net, save_dir='checkpoint'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    def quick_save_():
        save_path = os.path.join(save_dir, net.__class__.__name__+'_quick_save')
        print('\tsaving to ' + save_path)
        with open(save_path, mode='bw') as modelfile:
            torch.save(net, modelfile)
    return quick_save_

def validation(net, X, y, criterion):
    def validation_():
        net_state=net.training
        net.eval()
        y_ = net(X)
        eval = criterion(y_,y)
        print('\tValidation: %4f' % eval)
        net.train(net_state)
    return validation_

def reboot(net, optimizer=None, **net_args):
    assert isinstance(net,nn.Module), 'cannot reboot on this instance, use a nn.Module'
    def reboot_():
        net.__init__(**net_args)
        if optimizer:
            optim_params = optimizer.state_dict()
            optimizer.__init__(net.parameters(), lr=0)
            optimizer.load_state_dict(optim_params)
        print('\t%s rebooted' % net.__class__.__name__)
    return reboot_



