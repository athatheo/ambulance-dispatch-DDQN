# Dispatching ambulances in the Netherlands withDeep Reinforcement Learning

To run the code:

- Step 1: Make sure the environment is set up correctly
```
from Environment import Environment
env = Environment()
env.import_data()
```

- Step 2: Configure hyperparameters. In `deep_q_learning_main.py` you can configure number of episodes and their length. In `Learner.py` you can configure the batch size and in `Memory.py` the memory size. In `Model.py` you can configure the learning rate, and in `QNet.py` you can configure the layer  nodes.
- Step 3: Run the `deep_q_learning_main.py`

If you have an already made version of the model, you can load it using the `shelve` module.

# Project Report

The report for this project can be found here.

# Information

Group project created in the context of TU Delft's CS4320TU Applied AI Project.

Team 12:

Francesca Drummer

Thomas Georgiou

Athanasios Theocharis

Silvana van der Voort
