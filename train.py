import os
import time
import json
import argparse
import logging
import signal
import sys
from pathlib import Path
import random
from types import SimpleNamespace
import faulthandler
from pytorch_optimizer import Ranger21
from torchvision.transforms import v2

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from timm.layers import trunc_normal_, DropPath

from datasets import load_dataset
from torchvision.transforms import v2

from huggingface_hub import HfApi
from huggingface_hub.utils import LocalTokenNotFoundError
import yt_dlp
from yt_dlp.utils import DownloadError, ExtractorError

#from decord import VideoReader
#from decord.bridge import set_bridge
from torchcodec.decoders import VideoDecoder
from torchcodec.transforms import Resize, CenterCrop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.models import boundary_transformer_base

import timm
from timm.loss import JsdCrossEntropy
#import models
import torch.nn as nn


DATA_SYNC_ID = '100478946840763112718'

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

train_transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.RandomHorizontalFlip(),
        #v2.RandomApply([
        #    v2.ColorJitter(0.4, 0.4, 0.4, 0.1)
        #], p=0.8),
        #v2.RandomGrayscale(p=0.2),
        #v2.Normalize(mean=[0.54, 0.5, 0.474], std=[0.234, 0.235, 0.231])
    ])

#used to initialize head weights
def initialize_weights(m):
    if isinstance(m, (nn.Linear)):
        trunc_normal_(m.weight, std=.02)
        nn.init.constant_(m.bias, 0)

def process_youtube_videos(batch, transform=None):
    #print("Spawned thread")
    ydl_opts = {
        'cookiefile': 'premium_cookies.txt',
        'format': 'bestvideo[ext=mp4]/mp4',  
        'js_runtimes': { 'node': {'path': None} }, 
        'extractor_args': {
            'youtube': {
                'player_skip': ['webpage', 'configs'],
                #'player_client': ['web_safari', 'web_creator']
            }
        },
        'quiet': True,
        'no_warnings': True,
        'skip_download': True
    }

    label_window = 16
    clip_size = 256
    features = []
    labels = []
   
    batch_uids = batch['video_uid']
    batch_nodes = batch['nodes']
    for uid, nodes in zip(batch_uids, batch_nodes):
        try:
            # 1. Get the direct stream URL
            url = f'https://www.youtube.com/watch?v={uid}'
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                direct_stream_url = info['url']
            
            #fps = info.get("fps")
            w = int(info.get("width"))
            h = int(info.get("height"))

            target_size = 256
            if w < h:
                new_w = target_size
                new_h = int(h * (target_size / w))
            else:
                new_h = target_size
                new_w = int(w * (target_size / h))

            # 2. Decode the video stream into frames
            decoder = VideoDecoder(
                direct_stream_url, 
                device="cpu", 
                seek_mode="approximate",  
                transforms=[
                    Resize(size=(new_h, new_w)),  # Resizes smaller edge to 256
                    CenterCrop(size=(224, 224))   # Crops the center to 224x224
                ]
            )

            total_frames = decoder.metadata.num_frames
            fps = decoder.metadata.average_fps
            remainder = total_frames % clip_size
        
            num_windows = total_frames // label_window
            window_labels = torch.zeros(num_windows)

            transition_frames = []
            # Extract transition frames 
            for node in nodes:
                # Filter by your specific hierarchical level
                start_time = node["start"]
                end_time = node["end"]
                
                if end_time - start_time > 4.0: 
                    start_frame = int(round(start_time * fps))
                    end_frame = int(round(end_time * fps))
                    
                    transition_frames.append(start_frame)
                    transition_frames.append(end_frame)
            
            transition_frames = list(set(transition_frames))

            # Generate non-overlapping windows
            for start_frame in range(0, total_frames - (total_frames % label_window), label_window):
                end_frame = min(start_frame + label_window - 1, total_frames - 1)
                
                # Check if any transition falls within this window
                has_transition = any(start_frame <= t <= end_frame for t in transition_frames)
                window_label = 1 if has_transition else 0
                window_labels[start_frame // label_window] = window_label

            # 3. Extract Features
            for start_idx in range(0, total_frames-remainder, clip_size):
                end_idx = min(start_idx + clip_size, total_frames)
                feature = decoder.get_frames_in_range(start=start_idx, stop=end_idx).data
                feature = train_transform(feature)
                label = window_labels[int(start_idx/label_window) : int(end_idx/label_window)]

                features.append(feature)
                labels.append(label)
            
        except ExtractorError as e:
            print(f"Error processing {url}: {e}")
            continue
        except DownloadError as e:
            print(f"Error processing {url}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue
    
    
    return {"features": features, "labels": labels}


def custom_collate_fn(batch):
    batch_features = torch.stack([d['features'] for d in batch]).transpose(1, 2)
    batch_labels = torch.stack([d['labels'] for d in batch])
    return batch_features, batch_labels


def set_loader(config):
    '''
    Validate hugginface token
    '''
    
    try:
        api = HfApi(token=token)
        user_info = api.whoami()
        print(f"Token is valid! Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Invalid token or authentication failed: {e}")

    dataset = load_dataset(
        "parquet",
        data_files=f"hf://datasets/facebook/Action100M-preview/data/*.parquet",
        streaming=True,
        token=token
    )

    train_dataset = dataset['train']
    
    train_dataset_new = train_dataset.map(
        process_youtube_videos, 
        batched=True, 
        batch_size=2,
        remove_columns=train_dataset.column_names,
    )

    print(f"Number of shards: {train_dataset_new.n_shards}") 

    # Create data loaders
    train_loader = DataLoader(
        train_dataset_new,
        batch_size=4,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    return train_loader
    

def set_model():
    #print(f"Device count: {torch.cuda.device_count()}")
    model = boundary_transformer_base()
    #if torch.cuda.device_count() > 1:
    #    print(f"{torch.cuda.device_count()} devices detected, utilizing parallel training")
    #    model.encoder = torch.nn.DataParallel(model.encoder)
    model = model.to('cuda')
    return model

def set_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

class UniversalTrainer:
    """
    Universal trainer class that works with modular training configurations
    Supports multiple model architectures and background execution
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda")
        self.best_val_loss = float("inf")
        self.train_step = 0
        self.should_stop = False
        
        # Setup signal handlers for graceful shutdown
        self.setup_signal_handlers()

        # Setup logging
        self.setup_logging()
    
        #get model
        self.model = set_model()

        #for name, param in self.model.named_parameters():
        #    print(f"Layer: {name} | Size: {param.size()} | Trainable: {param.requires_grad}")

        #get loss function
        self.criterion = set_criterion()
    
        #set up optimizer
        self.lr = 4e-3
        num_epochs=30
        batches_per_epoch=197
        #self.optimizer = Ranger21(params=self.model.parameters(), lr=self.lr, num_iterations=num_epochs*batches_per_epoch)
        self.optimizer = torch.optim.AdamW(
           params=self.model.parameters(), lr=self.lr, weight_decay=0.01
        )

        #set up scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer=self.optimizer, 
            T_0=10000,   # Number of iterations/steps for the first restart
            T_mult=1     # Factor to increase T_0 after each restart (1 = no increase)
        )
        # Mixed precision training
        self.config.mixed_precision = False

        self.scaler = GradScaler() if self.config.mixed_precision else None
        
        train_loader = set_loader(config)

        self.train_loader = train_loader

        #for batch_idx, (inputs, targets) in enumerate(train_loader):
        #    print(f"Batch {batch_idx}:")
        #    print(f"  Inputs shape: {inputs.shape}")
        #    print(f" Targets shape: {targets.shape}")
            #print(f"  Inputs: \n{inputs}")
            #print(f"  Targets: {targets}\n")
        #self.logger.info(f"📊 Training clips: {train_info['total_clips']:,}")
        #self.logger.info(f"📊 Validation clips: {val_info['total_clips']:,}")

    def _override_config_with_cli_args(self, config):
        """Override default config with CLI arguments (CLI args take precedence)"""
        # Override training parameters if provided via CLI
        if hasattr(config, "epochs") and config.epochs is not None:
            self.training_config["epochs"] = config.epochs
        if hasattr(config, "batch_size") and config.batch_size is not None:
            self.training_config["batch_size"] = config.batch_size
        if (
            hasattr(config, "val_batch_size")
            and config.val_batch_size is not None
        ):
            self.training_config["val_batch_size"] = config.val_batch_size
        if hasattr(config, "backbone_lr") and config.backbone_lr is not None:
            self.training_config["backbone_lr"] = config.backbone_lr
        if hasattr(config, "head_lr") and config.head_lr is not None:
            self.training_config["head_lr"] = config.head_lr
        if hasattr(config, "weight_decay") and config.weight_decay is not None:
            self.training_config["weight_decay"] = config.weight_decay
        if hasattr(config, "grad_clip") and config.grad_clip is not None:
            self.training_config["grad_clip"] = config.grad_clip
        if hasattr(config, "num_workers") and config.num_workers is not None:
            self.training_config["num_workers"] = config.num_workers
        if hasattr(config, "cache_size") and config.cache_size is not None:
            self.training_config["cache_size"] = config.cache_size
        if (
            hasattr(config, "mixed_precision")
            and config.mixed_precision is not None
        ):
            self.training_config["mixed_precision"] = config.mixed_precision
        if hasattr(config, "save_freq") and config.save_freq is not None:
            self.training_config["save_freq"] = config.save_freq

        # Override model parameters if provided
        if hasattr(config, "n_segment") and config.n_segment is not None:
            self.training_config["n_segment"] = config.n_segment
        if hasattr(config, "num_actions") and config.num_actions is not None:
            self.training_config["num_actions"] = config.num_actions
        if hasattr(config, "history_len") and config.history_len is not None:
            self.training_config["history_len"] = config.history_len
        if (
            hasattr(config, "unfreeze_last_n_layers")
            and config.unfreeze_last_n_layers is not None
        ):
            self.training_config["unfreeze_last_n_layers"] = (
                config.unfreeze_last_n_layers
            )

        # Override loss-specific parameters if provided
        if (
            hasattr(config, "lambda_action")
            and config.lambda_action is not None
        ):
            self.training_config["lambda_action"] = config.lambda_action
        if hasattr(config, "focal_beta") and config.focal_beta is not None:
            self.training_config["focal_beta"] = config.focal_beta
        if hasattr(config, "focal_gamma") and config.focal_gamma is not None:
            self.training_config["focal_gamma"] = config.focal_gamma
        if (
            hasattr(config, "label_smoothing")
            and config.label_smoothing is not None
        ):
            self.training_config["label_smoothing"] = config.label_smoothing

        # Override scheduler parameters if provided
        if hasattr(config, "scheduler_t0") and config.scheduler_t0 is not None:
            self.training_config["scheduler_t0"] = config.scheduler_t0
        if hasattr(config, "min_lr") and config.min_lr is not None:
            self.training_config["min_lr"] = config.min_lr

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""

        def signal_handler(signum, frame):
            signal_name = signal.Signals(signum).name
            print(
                f"\n🛑 Received {signal_name} signal. Initiating graceful shutdown..."
            )
            self.should_stop = True

            # Save current state immediately
            if hasattr(self, "logger"):
                self.logger.info(
                    f"🛑 Received {signal_name} signal. Saving checkpoint and shutting down..."
                )

        # Handle common termination signals
        signal.signal(signal.SIGTERM, signal_handler)  # Termination request
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C

        # Handle SSH disconnection (SIGHUP)
        def sighup_handler(signum, frame):
            if hasattr(self, "logger"):
                self.logger.info(
                    "🔌 SSH session disconnected (SIGHUP). Continuing training in background..."
                )
            # Don't stop training on SIGHUP - continue in background

        signal.signal(signal.SIGHUP, sighup_handler)

    def setup_logging(self):
        """Setup logging configuration if not already configured"""
        # Check if logging is already configured (by main function)
        if logging.getLogger().handlers:
            self.logger = logging.getLogger(__name__)
            self.logger.info("📝 Using existing logging configuration")
            return

        # Fallback logging setup (shouldn't be needed with new structure)
        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    log_dir / f"train_{int(time.time())}.log", mode="a"
                ),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("📝 Fallback logging configuration initialized")


    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        for batch_idx, (features, labels) in enumerate(self.train_loader):
            # Check for stop signal
            if self.should_stop:
                self.logger.info(
                    "🛑 Stop signal received. Finishing current epoch and saving checkpoint..."
                )
                break

            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"📈 Training batch {batch_idx} - Running avg loss: {total_loss / (batch_idx + 1) if batch_idx > 0 else 0:.4f}"
                )
                # Force log flush
                for handler in self.logger.handlers:
                    if hasattr(handler, "flush"):
                        handler.flush()

            try:
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with mixed precision using architecture-specific
                # logic
                if self.scaler:
                    with autocast(device_type="cuda"):
                        features = self.model(images)
                        features = features.view(-1, t, *features.shape[1:])
                        features = torch.mean(features, dim=1) # or reshaped_x.mean(dim=1)
                        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                        loss = self.criterion(features, labels)

                    # Backward pass
                    self.scaler.scale(loss).backward()
                    #loss.backward()

                    # Gradient clipping
                    if self.config.grad_clip is not None and self.config.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(features).permute(0, 2, 1)
                    labels = labels.long()
                    print(outputs.shape)
                    print(labels.shape)
                    input("Getting output shape")

                    loss = self.criterion(outputs, labels)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    if self.config.grad_clip is not None and self.config.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.grad_clip
                        )

                    self.optimizer.step()
                print(f"{batch_idx}: {loss.item()}")
                pred_actions = outputs.argmax(dim=1)
                true_actions = labels
                action_accuracy += (
                    (pred_actions == true_actions).float().mean().item()
                )
                print(f"{batch_idx}: {action_accuracy}")
                # Accumulate losses (uniform API)
                total_loss += loss.item()
                #print("Total loss:")

                # Log detailed batch info every 100 batches
                if batch_idx % 100 == 0:
                    #current_lr = self.optimizer.current_lr
                    #for param_group in self.optimizer.param_groups:
                    #    current_lr = param_group['lr']
                    # Build loss component string based on what's available
                    loss_parts = [f"Loss={loss.item():.4f}"]
                    #loss_parts.append(f"LR={current_lr:.2e}")
                    self.logger.info(
                        f"   📊 Batch {batch_idx}: {', '.join(loss_parts)}"
                    )

                self.train_step += 1

            except Exception as e:
                self.logger.error(f"❌ Error in batch {batch_idx}: {e}")
                self.logger.error(
                    f"   - frames shape: {batch['frames'].shape if 'frames' in batch else 'missing'}"
                )
                raise e

        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)

        # Log epoch completion
        self.logger.info("✅ Training epoch completed:")
        self.logger.info(f"   - Average loss: {avg_loss:.4f}")

        # Force log flush
        for handler in self.logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        # Build return dict based on available losses
        result = {"loss": avg_loss}
        print("Total training loss: ", total_loss)
        return result


    def load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint"""
        self.logger.info(f"🔄 Loading checkpoint from {checkpoint_path}")

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("✅ Model state restored")

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.logger.info("✅ Optimizer state restored")

        # Restore scheduler state
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.logger.info("✅ Scheduler state restored")

        # Restore scaler state if using mixed precision
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.logger.info("✅ Scaler state restored")

        # Restore training step counter
        self.train_step = checkpoint.get("train_step", 0)

        # Get starting epoch
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        self.logger.info("✅ Checkpoint loaded successfully:")
        self.logger.info(f"   - Resuming from epoch: {start_epoch}")
        self.logger.info(f"   - Training step: {self.train_step}")
        self.logger.info(f"   - Best validation loss: {best_val_loss:.4f}")

        return start_epoch, best_val_loss

    def save_checkpoint(self, epoch, val_results, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            #"scheduler_state_dict": self.scheduler.state_dict(),
            "val_results": val_results,
            "config": vars(self.config),
            "train_step": self.train_step,
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"💾 Best model saved: {best_path}")

        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        self.logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

    def train(self, resume_checkpoint=None):
        """Main training loop with graceful shutdown support"""
        best_val_loss = float("inf")
        start_epoch = 0

        # Load checkpoint if resuming
        if resume_checkpoint:
            try:
                start_epoch, best_val_loss = self.load_checkpoint(
                    resume_checkpoint
                )
                self.best_val_loss = best_val_loss  # Update instance variable
                self.logger.info(
                    f"🔄 Resumed training from epoch {start_epoch}"
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to load checkpoint: {e}")
                self.logger.info("🆕 Starting fresh training instead")
                start_epoch = 0
                self.best_val_loss = float("inf")

        self.logger.info("🚀 Starting training...")
        self.logger.info(
            f"🎯 Model: {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        self.logger.info(
            "📌 Training supports background execution - safe to disconnect SSH"
        )

        try:
            for epoch in range(start_epoch, self.config.epochs):
                # Check for stop signal at start of epoch
                if self.should_stop:
                    self.logger.info(
                        "🛑 Stop signal received. Saving final checkpoint..."
                    )
                    break

                epoch_start_time = time.time()
                self.logger.info(
                    f"🎯 Starting Epoch {epoch + 1}/{self.config.epochs}"
                )

                # Train
                print("Epoch: ", epoch)
                train_results = self.train_epoch()

                # Check again after training epoch
                if self.should_stop:
                    self.logger.info(
                        "🛑 Stop signal received after training epoch. Saving checkpoint..."
                    )
                    # Save current state before exiting
                    self.save_checkpoint(
                        epoch + 1, {"loss": float("inf")}, is_best=False
                    )
                    break

                val_results = 0.0
                is_best = 0.0
                epoch_time = time.time() - epoch_start_time

                # Log epoch results
                # Build progress log string based on available metrics
                log_parts = [
                    f"Epoch {epoch + 1:3d}/{self.config.epochs}",
                    f"Train Loss: {train_results['loss']:.4f}",
                ]
                log_parts.extend(
                    [f"Time: {epoch_time:.1f}s", f"{'🌟' if is_best else ''}"]
                )

                self.logger.info(" | ".join(log_parts))

                # Save checkpoint
                if (epoch + 1) % self.config.save_freq == 0 or is_best:
                    self.save_checkpoint(epoch + 1, val_results, is_best)

                # Check for stop signal after checkpoint
                if self.should_stop:
                    self.logger.info(
                        "🛑 Stop signal received after checkpoint. Exiting gracefully..."
                    )
                    break

            if not self.should_stop:
                self.logger.info("✅ Training completed successfully!")
            else:
                self.logger.info(
                    "🛑 Training stopped gracefully by user signal"
                )

        except KeyboardInterrupt:
            self.logger.info(
                "🛑 Training interrupted by KeyboardInterrupt. Saving checkpoint..."
            )
            try:
                self.save_checkpoint(
                    epoch + 1,
                    val_results
                    if "val_results" in locals()
                    else {"loss": float("inf")},
                    is_best=False,
                )
            except Exception as e:
                self.logger.error(f"❌ Failed to save final checkpoint: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Training failed with error: {e}")
            try:
                self.save_checkpoint(
                    epoch + 1, {"loss": float("inf")}, is_best=False
                )
                self.logger.info("💾 Emergency checkpoint saved")
            except Exception as e:
                self.logger.error(
                    f"❌ Failed to save emergency checkpoint: {e}"
                )
            raise


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

    # Model configuration
    parser.add_argument(
        "--n-segment", type=int, default=8, help="Number of video frames"
    )
    parser.add_argument(
        "--num-actions", type=int, default=14, help="Number of action classes"
    )
    parser.add_argument(
        "--history-len",
        type=int,
        help="History length (completion or action probabilities)",
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

    return args


def daemonize():
    """Detach from terminal and run in background"""
    try:
        # First fork
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            sys.exit(0)
    except OSError as e:
        print(f"❌ Fork #1 failed: {e}")
        sys.exit(1)

    # Decouple from parent environment - but stay in current directory
    os.setsid()
    os.umask(0)

    try:
        # Second fork
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            sys.exit(0)
    except OSError as e:
        print(f"❌ Fork #2 failed: {e}")
        sys.exit(1)

    # Redirect standard file descriptors
    with open("/dev/null", "r") as devnull_r:
        os.dup2(devnull_r.fileno(), sys.stdin.fileno())

    # Don't redirect stdout/stderr - let logging handle output
    # This allows the log files to capture all output properly


def setup_early_logging(config):
    """Setup logging before daemonization"""
    output_dir = Path(config.output_dir)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"train_{int(time.time())}.log"

    # Configure logging with immediate flush for daemon mode
    handlers = [logging.FileHandler(log_file, mode="a")]

    # Add console handler only if not in background
    if not config.background:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )

    # Force flush after each log
    for handler in logging.getLogger().handlers:
        if hasattr(handler, "flush"):
            handler.flush()

    return logging.getLogger(__name__), str(log_file)


def main():
    """Main training function with background execution support"""
    # Create configuration
    config = create_config()

    # Create output directory and setup logging BEFORE daemonizing
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = str(output_dir.absolute())

    # Setup logging before any daemonization
    logger, log_file_path = setup_early_logging(config)

    # Set random seed (after potential fork)
    set_seed(config.seed)

    config.device == "cuda"

    # Create output directory (if not already created in background mode)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config (convert Path objects to strings for JSON serialization)
    config_dict = vars(config).copy()
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_dict[key] = str(value)

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Write PID file for background processes
    if config.background:
        pid_file = output_dir / "training.pid"
        with open(pid_file, "w") as f:
            f.write(str(os.getpid()))
        print(f"📝 PID file created: {pid_file}")


    # Create trainer and start training
    trainer = UniversalTrainer(config)
    trainer.train(resume_checkpoint=config.resume)

    # Clean up PID file on normal exit
    if config.background:
        pid_file = output_dir / "training.pid"
        if pid_file.exists():
            pid_file.unlink()


if __name__ == "__main__":
    main()
