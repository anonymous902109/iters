# Interactive Reward Shaping  

The code for the paper ``Iterative Reward Shaping using Human Feedback for Correcting Reward Misspecification''.

## Installation  

```shell
conda create -n irs python=3.7  
conda activate irs  
pip install -r requirements.txt  
```

## Running Experiments  

At the moment, experiments can be run with simulated users.

```shell
python main.py --task gridworld  

python main.py --task highway  

python main.py --task inventory
```
