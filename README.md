This is the official repository for Robotic Fall Prediction with Egocentric Vision and Proprioception.

# Environment
Please refer to the environment.yml for dependecies.

# How to generate simulated data
1. data_generation.py is intended for data generation.

2. Option --number indicates the **TOTAL** number of episodes. Better give it an even number. Otherwise, the following data preprocessing might throw errors.

3. You may split data generation in multiple steps. Use --base to indicate where to start by entering the index of existing episodes. Otherwise, data_generation.py may throw errors or overwrite existing episodes.

4. The maximum episodes for each category (fall and nonfall) is 5000. We do not offer any options in the argument parser for this parameter. Please refere to line 345 and 364 if you want to modify the maximum value.

# How to run this project/code
1. We provide a main.py file under folders "ours", "baseline_siMLPe" and "baseline_Egofall". Please refer to it for either training and testing or testing only. We also provide commnad examples in the main.py. 

2. Option --dataset only supports our real ("RP") and simulated ("VP") datases.

3. Use --annotation to customize the name of your experiments.

4. Please uncomment the five if branches in main.py files and carefully input the modules for option --model if you want to run an ablation study.

# Model weights 
Please refer to https://drive.google.com/drive/folders/1vqvRzSSnLAAFpO4HIVXgFujRzfXtPiRb?usp=drive_link for trained models/modules.

# Miscellaneous
Code for tables and plots are provided in the folder named "utils".
