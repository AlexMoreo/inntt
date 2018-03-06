# inntt: Interactive NeuralNet Trainer for pyTorch

Finding the right hyperparameters when training deep learning models can be painful. The practitioner often ends up applying a  trial/error approach to set them, based on the observation of some indicators (tr_loss, val_loss, etc.). Each little modification typically entails retraining from scratch. Interactive NeuralNet Trainer for pyTorch (INNTT) allows you to modify many parameters <b>on the fly</b>, interacting with the keyboard. 

Some routines/features currently supported:
- increase/decrease the learning rate (actually, any optimizer parameter such as the weight_decay, etc)
- invoke the validation phase
- quick-load/save the model: this allows you to safely experiment with different parameters (if something goes wrong, simply quick-load your model)
- reboot the model parameters (useful when gradients explode and create nan parameters, or simply when you want to retrain from scratch but with the, say, lr, weight_decay, and drop_prob you have found to work best)
- logging of the interactions

The inntt works currently only on the Linux's terminal; i'll fix that. 
I would also like to show some examples with growing-nets, this time controlled by the user.

