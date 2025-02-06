hyperparameters_virtual={
    # datasets parameters
    'dataset_name': 'vir_poppy',
    'N': 60,
    'num_samples': 100, # an even number LEQ the size of your dataset for a quick run, otherwise, None for using the whole dataset
    'bs': 64,
    'k_fold': 10,
    'prediction_span': [10, 20, 30, 40, 50, 60],
}

hyperparameters_real={
    # datasets parameters
    'dataset_name': 'real_poppy',
    'bs': 64,
    'N': 24,
    'k_fold': 10,
    'prediction_span': [4, 8, 12, 16, 20, 24],
}
