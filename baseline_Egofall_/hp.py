hyperparameters_virtual={
    # datasets parameters
    'dataset_name': 'vir_poppy',
    'num_samples': 10000, # an even number LEQ the size of your dataset for a quick run, otherwise, None for using the whole dataset
    'bs': 64,
    'k_fold': 10,
}

hyperparameters_real={
    # datasets parameters
    'dataset_name': 'real_poppy',
    'bs': 8,
    'k_fold': 10,
}
