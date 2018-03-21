from distutils.core import setup

setup(
    name='Interactive neural net trainer for Torch',
    version='1.0',
    scripts=['inntt.py'],
    url='https://github.com/AlexMoreo/inntt',
    license='',
    author='Alejandro Moreo Fernandez',
    author_email='alejandro.moreo@isti.cnr.it',
    description='Allows to modify on the fly some learned parameters (e.g., the learning rate, weight decay) when training a neural network. '
)

