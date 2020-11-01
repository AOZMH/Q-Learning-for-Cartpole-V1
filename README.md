# Q-Learning-for-Cartpole-V1
Naive implementation of q-learning on Carpole-v1 in openai-gym

## Execution
* Training

Configurate the value of parameter **test_or_eval** at the bottom of **main.py** to **'train'**, set up other hyper parameters.
> python main.py

* Evaluating

To test the rate at which the model can survive no less than 200 steps.
Configurate the value of parameter **test_or_eval** of **main.py** to **'eval'**, fill in the route to the well-trained q-table to be evaluated to **checkpoint_q_table**, set **num_trials** to the number of runs when evaluating the model.
> python main.py

* Illustrating

To continuously run one episode until the pole falls down or the cart moves away and illustrate the process on a window, no early stopping on 200 steps.
Configurate the parameter **checkpoint_q_table** of **test_and_illustrate.py** to the q-table file to be tested (e.g. './data/q_table_02lr.npy'), guarantee that the state numbers accords with the q-table.
> python test_and_illustrate.py
