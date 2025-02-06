hyperparameters_virtual={
    # datasets parameters
    'splits': [.8, .1, .1],
    'dataset_name': 'vir_poppy',
    'data_type': 'JA',
    'num_samples': None, # an even number LEQ the size of your dataset for a quick run, otherwise, None for using the whole dataset
    
    # siMLPe
    'bs_siMLPe': 64,
    # model parameters
    'N': 60, # the length of input sequences
    'prediction_span': [10, 20, 30, 40, 50, 60], # the length of output sequences
    'motion_dim': 42, # the number of joints
    'lr': 3e-3,
    'weight_decay': 1e-4,
    'total_iterations': 10000,

    # Popeyes
    'bs_Popeyes': 32,
    # model parameters
    'img_height': 128,
    'img_width': 128,
    'latent_size': 256,
    'lr_Popeyes': 1e-4,
    'weight_decay_Popeyes': 5e-4,
    'total_iterations_Popeyes': 10000,

    # Traj
    'bs_Traj': 64,
    # model parameters
    'motion_dim_Traj': 42,
    'lr_Traj': 5e-7,
    'weight_decay_Traj': 1e-5,
    'total_iterations_Traj': 10000
}

hyperparameters_real={
    # datasets parameters
    'splits': [.8, .1, .1],
    'dataset_name': 'real_poppy',
    'data_type': 'JA',
    'num_samples': None, # An even number LEQ the size of your dataset for a quick run, otherwise, None for using the whole dataset
    
    # siMLPe
    'bs_siMLPe': 256,
    # model parameters
    'N': 24, # the length of input sequences
    'prediction_span': [4, 8, 12, 16, 20, 24], # the length of output sequences
    'motion_dim': 25, # the number of joints
    'lr': 3e-5,
    'weight_decay': 1e-4,
    'total_iterations': 10000,

    # Popeyes
    'bs_Popeyes': 16,
    # model parameters
    'img_height': 96,
    'img_width': 128,
    'latent_size': 256,
    'lr_Popeyes': 1e-4,
    'weight_decay_Popeyes': 1e-4,
    'total_iterations_Popeyes': 10000,

    # Traj
    'bs_Traj': 64,
    # model parameters
    'motion_dim_Traj': 25,
    'lr_Traj': 5e-7,
    'weight_decay_Traj': 1e-5,
    'total_iterations_Traj': 10000
}