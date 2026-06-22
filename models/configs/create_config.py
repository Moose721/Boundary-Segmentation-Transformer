def create_config():
    """Create training configuration"""
    parser = argparse.ArgumentParser(
        description="Train Dual-Branch Manufacturing Model"
    )

    # Dataset paths - only split files supported for performance
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Directory containing split dataset subdirectories",
    )
    parser.add_argument(
        "--train-files",
        type=str,
        default="/mnt/shared/gpfs/home/allenp2/FaiProject/large_dataset3/splits/train_files.txt",
        help="Path to train file list",
    )
    parser.add_argument(
        "--val-files",
        type=str,
        default="/mnt/shared/gpfs/home/allenp2/FaiProject/large_dataset3/splits/val_files.txt",
        help="Path to validation file list",
    )

    # Architecture selection
    parser.add_argument(
        "--arch",
        type=str,
        default="mobilenet_tsm_vis_hist",
        help="Architecture to use (see --list-archs)",
    )
    parser.add_argument(
        "--list-archs",
        action="store_true",
        help="List available architectures and exit",
    )
    
    parser.add_argument(
        "--unfreeze-last-n-layers",
        type=int,
        help="Number of backbone layers to unfreeze",
    )

    # Training configuration
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=64, help="Validation batch size"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers",
    )

    # Optimizer configuration
    parser.add_argument(
        "--backbone-lr", type=float, help="Learning rate for backbone layers"
    )
    parser.add_argument(
        "--head-lr", type=float, help="Learning rate for head layers"
    )
    parser.add_argument("--weight-decay", type=float, help="Weight decay")
    parser.add_argument(
        "--grad-clip", type=float, help="Gradient clipping norm"
    )

    # Loss configuration (model-specific)
    parser.add_argument(
        "--lambda-action",
        type=float,
        help="Weight for action classification loss (dual-branch models)",
    )
    parser.add_argument(
        "--focal-beta",
        type=float,
        help="Focal-R beta parameter (dual-branch models)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        help="Focal-R gamma parameter (dual-branch models)",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing for action classification (action-only models)",
    )

    # Scheduler configuration
    parser.add_argument(
        "--scheduler-t0", type=int, help="Cosine annealing restart period"
    )
    parser.add_argument("--min-lr", type=float, help="Minimum learning rate")

    # Training options
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        default=True,
        help="Pin memory for data loaders",
    )
    parser.add_argument(
        "--cache-size", type=int, default=1000, help="Dataset cache size"
    )

    # Logging and saving
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="manufacturing_model",
        help="Experiment name",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training (auto, cuda, cpu)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Background execution
    parser.add_argument(
        "--background",
        action="store_true",
        default=True,
        help="Run in background mode (detach from terminal)",
    )
    parser.add_argument(
        "--no-background",
        dest="background",
        action="store_false",
        help="Disable background mode (run in foreground)",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training from",
    )

    args = parser.parse_args()

    # Handle --list-archs flag
    if args.list_archs:
        print("\n📋 Available training architectures:")
        print("=" * 50)
        for arch_name in get_available_configs():
            info = get_config_info(arch_name)
            print(f"\n📦 {arch_name}")
            if "error" in info:
                print(f"   Error: {info['error']}")
            else:
                print(
                    f"   Description: {info.get('description', 'No description').strip()}"
                )
                config = info.get("default_config", {})
                print(f"   Epochs: {config.get('epochs', 'N/A')}")
                print(f"   Batch size: {config.get('batch_size', 'N/A')}")
                print(
                    f"   Data requirements: {list(k for k, v in info.get('data_requirements', {}).items() if v)}"
                )
        print("\n💡 Use --arch <name> to select an architecture")
        sys.exit(0)

    # Set up paths - only support split files for better performance
    base_dir = Path(__file__).parent

 
    if args.train_files and args.val_files:
        # Use explicitly provided split file lists
        args.train_files_path = base_dir / args.train_files
        args.val_files_path = base_dir / args.val_files
    else:
        # Auto-detect split files in standard locations
        dataset_dir = base_dir / args.dataset_dir
        args.train_files_path = dataset_dir / "train_splits" / "train_files.txt"
        args.val_files_path = dataset_dir / "val_splits" / "val_files.txt"



    return args