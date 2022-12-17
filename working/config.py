

class CFG:
    """
    Parameters used for training
    """
    #debug
    debug = False
    train = True
    train_fold = [0,1]
    num_workers = 4

    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Images
    size = 256
    size = 512

    # k-fold
    n_fold = 4  # Stratified GKF

    # Model
    model = "efficientnet_b1"
    exp = 'baseline'
    pretrained_weights = None
    num_classes = 1
    n_channels = 3

    # Training    
    loss_config = {
        "name": "bce",  # dice, ce, bce
        "smoothing": 0.,  # 0.01
        "activation": "sigmoid",  # "sigmoid", "softmax"
        "aux_loss_weight": 0,
    }

    data_config = {
        "batch_size": 32,
        "val_bs": 48,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 3e-4,
        "warmup_prop": 0.1,
        "betas": (0.9, 0.999),
        "eps":1e-6,
        "max_grad_norm": 10.,
    }
    num_cycles=0.5
    num_warmup_steps=0
    llrd = False
    epochs = 4
    apex = True
    batch_scheduler = True
    scheduler = 'cosine'
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    print_freq = 20
    wandb = False
    
    ## Other stuff
    # Augmentations : Only HorizontalFlip