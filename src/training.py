import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable, List, Union
import time
import os
import copy
import json
import itertools
import random
import numpy as np
from dataclasses import dataclass

from src.metrics import get_loss_function

class TrainingConfig:
    """
    Module 10: Configuration class for training hyperparameters.
    Centralizes all training settings for reproducibility.
    """
    def __init__(
        self,
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        optimizer_name: str = "adamw",
        num_epochs: int = 50,
        device: str = "auto",
        mixed_precision: bool = True,
        gradient_clip_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        # Learning rate scheduling
        scheduler_name: str = "cosine",
        scheduler_params: Optional[Dict] = None,
        warmup_epochs: int = 0,
        # Early stopping
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 1e-4,
        early_stopping_monitor: str = "val_loss",
        early_stopping_mode: str = "min",
        # Checkpointing
        save_top_k: int = 3,
        save_last: bool = True,
        # GPU optimizations
        compile_model: bool = False,
        channels_last: bool = False,
        # Hyperparameter tuning
        hp_search_space: Optional[Dict] = None,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name.lower()
        self.num_epochs = num_epochs
        self.device = self._resolve_device(device)
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        # Scheduler
        self.scheduler_name = scheduler_name.lower()
        self.scheduler_params = scheduler_params or {}
        self.warmup_epochs = warmup_epochs
        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_monitor = early_stopping_monitor
        self.early_stopping_mode = early_stopping_mode
        # Checkpointing
        self.save_top_k = save_top_k
        self.save_last = save_last
        # GPU optimizations
        self.compile_model = compile_model and self.device.type == "cuda"
        self.channels_last = channels_last and self.device.type == "cuda"
        # Hyperparameter tuning
        self.hp_search_space = hp_search_space or {}

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def __repr__(self):
        return (f"TrainingConfig(batch_size={self.batch_size}, lr={self.learning_rate}, "
                f"optimizer={self.optimizer_name}, epochs={self.num_epochs}, "
                f"device={self.device}, mixed_precision={self.mixed_precision}, "
                f"scheduler={self.scheduler_name}, early_stop_patience={self.early_stopping_patience})")


def get_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """
    Module 10: Factory function to create optimizers.
    Supports Adam, AdamW, SGD with momentum, and RMSprop.
    """
    params = model.parameters()

    if config.optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    elif config.optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
    elif config.optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_name}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig, num_training_steps: int):
    """
    Module 10: Factory function to create learning rate schedulers.
    
    Supported schedulers:
    - cosine: CosineAnnealingLR with optional warmup
    - cosine_warm_restarts: CosineAnnealingWarmRestarts
    - step: StepLR
    - multistep: MultiStepLR
    - exponential: ExponentialLR
    - reduce_on_plateau: ReduceLROnPlateau
    - onecycle: OneCycleLR
    - polynomial: PolynomialLR
    - constant: No scheduling (constant LR)
    """
    scheduler_name = config.scheduler_name
    params = config.scheduler_params
    warmup_epochs = config.warmup_epochs

    if scheduler_name == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - warmup_epochs,
            eta_min=params.get("eta_min", 1e-6),
        )
    elif scheduler_name == "cosine_warm_restarts":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=params.get("T_0", 10),
            T_mult=params.get("T_mult", 1),
            eta_min=params.get("eta_min", 1e-6),
        )
    elif scheduler_name == "step":
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("step_size", 10),
            gamma=params.get("gamma", 0.1),
        )
    elif scheduler_name == "multistep":
        main_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=params.get("milestones", [30, 60, 90]),
            gamma=params.get("gamma", 0.1),
        )
    elif scheduler_name == "exponential":
        main_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params.get("gamma", 0.95),
        )
    elif scheduler_name == "reduce_on_plateau":
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=params.get("mode", "min"),
            factor=params.get("factor", 0.5),
            patience=params.get("patience", 5),
            min_lr=params.get("min_lr", 1e-6),
        )
    elif scheduler_name == "onecycle":
        main_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=num_training_steps,
            pct_start=params.get("pct_start", 0.3),
            anneal_strategy=params.get("anneal_strategy", "cos"),
        )
    elif scheduler_name == "polynomial":
        main_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=num_training_steps - warmup_epochs,
            power=params.get("power", 1.0),
        )
    elif scheduler_name == "constant":
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # ReduceLROnPlateau cannot be used with SequentialLR
        if scheduler_name == "reduce_on_plateau":
            # Return a custom scheduler that handles warmup then plateau
            return _WarmupPlateauScheduler(optimizer, warmup_scheduler, main_scheduler, warmup_epochs)
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs],
        )
    return main_scheduler


class _WarmupPlateauScheduler:
    """Wrapper to handle warmup + ReduceLROnPlateau combination."""
    def __init__(self, optimizer, warmup_scheduler, plateau_scheduler, warmup_epochs):
        self.optimizer = optimizer
        self.warmup_scheduler = warmup_scheduler
        self.plateau_scheduler = plateau_scheduler
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.in_warmup = True
    
    def step(self, metric=None):
        if self.in_warmup:
            self.warmup_scheduler.step()
            self.current_epoch += 1
            if self.current_epoch >= self.warmup_epochs:
                self.in_warmup = False
        else:
            self.plateau_scheduler.step(metric)
    
    def get_last_lr(self):
        if self.in_warmup:
            return self.warmup_scheduler.get_last_lr()
        return self.plateau_scheduler.get_last_lr()
    
    def state_dict(self):
        return {
            "warmup": self.warmup_scheduler.state_dict(),
            "plateau": self.plateau_scheduler.state_dict(),
            "current_epoch": self.current_epoch,
            "in_warmup": self.in_warmup,
        }
    
    def load_state_dict(self, state_dict):
        self.warmup_scheduler.load_state_dict(state_dict["warmup"])
        self.plateau_scheduler.load_state_dict(state_dict["plateau"])
        self.current_epoch = state_dict["current_epoch"]
        self.in_warmup = state_dict["in_warmup"]


class Trainer:
    """
    Module 10: Comprehensive Training/Validation Pipeline.
    
    Features:
    - Configurable batch size, learning rate, optimizer (Adam/AdamW)
    - Learning rate scheduling (cosine, step, plateau, onecycle, etc.)
    - Early stopping with configurable patience and monitor metric
    - Mixed precision training (AMP) for faster GPU training
    - Gradient clipping for stability
    - Model compilation (torch.compile) for faster GPU execution
    - Channels-last memory format for better GPU utilization
    - Top-K checkpointing with automatic cleanup
    - Automatic device management
    - Detailed logging and metrics tracking
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config: TrainingConfig,
        metrics_fn: Optional[Callable] = None,
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.metrics_fn = metrics_fn

        # GPU optimizations
        if config.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if config.compile_model and hasattr(torch, "compile"):
            print("  [GPU Opt] Compiling model with torch.compile()...")
            self.model = torch.compile(self.model, mode="max-autotune")

        self.optimizer = get_optimizer(model, config)
        
        # Use new GradScaler API for PyTorch 2.0+
        if config.mixed_precision and config.device.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
        else:
            self.scaler = torch.amp.GradScaler('cpu', enabled=False)
        
        # Learning rate scheduler
        num_training_steps = config.num_epochs
        self.scheduler = get_scheduler(self.optimizer, config, num_training_steps)
        self.scheduler_is_plateau = config.scheduler_name == "reduce_on_plateau"
        
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_metrics": [], "val_metrics": [],
            "learning_rates": [], "epoch_times": []
        }
        
        # Early stopping state
        self.early_stop_counter = 0
        self.best_val_loss = float('inf')
        self.best_val_metric = -float('inf')
        self.current_epoch = 0
        
        # Top-K checkpoint tracking
        self.top_k_checkpoints: List[tuple] = []  # (metric, filename)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def train_epoch(self) -> Dict[str, float]:
        """Single training epoch with mixed precision support."""
        self.model.train()
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(self.train_loader)

        for batch_idx, (before, after, masks) in enumerate(self.train_loader):
            before = before.to(self.config.device, non_blocking=True)
            after = after.to(self.config.device, non_blocking=True)
            masks = masks.to(self.config.device, non_blocking=True)
            
            if self.config.channels_last:
                before = before.to(memory_format=torch.channels_last)
                after = after.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)

            self.optimizer.zero_grad(set_to_none=True)

            # Use new autocast API for PyTorch 2.0+
            with torch.amp.autocast('cuda' if self.config.device.type == 'cuda' else 'cpu', 
                                    enabled=self.config.mixed_precision):
                outputs = self.model(before, after)
                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()

            if self.config.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

            if self.metrics_fn:
                with torch.no_grad():
                    batch_metrics = self.metrics_fn(outputs.detach(), masks)
                    for k, v in batch_metrics.items():
                        total_metrics[k] = total_metrics.get(k, 0.0) + v

            if batch_idx % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  [Epoch {self.current_epoch+1}/{self.config.num_epochs}] "
                      f"Batch {batch_idx+1}/{num_batches} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()} if total_metrics else {}
        return {"loss": avg_loss, **avg_metrics}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = len(self.val_loader)

        for before, after, masks in self.val_loader:
            before = before.to(self.config.device, non_blocking=True)
            after = after.to(self.config.device, non_blocking=True)
            masks = masks.to(self.config.device, non_blocking=True)
            
            if self.config.channels_last:
                before = before.to(memory_format=torch.channels_last)
                after = after.to(memory_format=torch.channels_last)
                masks = masks.to(memory_format=torch.channels_last)

            with torch.amp.autocast('cuda' if self.config.device.type == 'cuda' else 'cpu',
                                    enabled=self.config.mixed_precision):
                outputs = self.model(before, after)
                loss = self.criterion(outputs, masks)

            total_loss += loss.item()

            if self.metrics_fn:
                batch_metrics = self.metrics_fn(outputs, masks)
                for k, v in batch_metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

        avg_loss = total_loss / num_batches
        avg_metrics = {k: v / num_batches for k, v in total_metrics.items()} if total_metrics else {}
        return {"loss": avg_loss, **avg_metrics}

    def _check_early_stopping(self, val_results: Dict[str, float]) -> bool:
        """Check early stopping condition. Returns True if training should stop."""
        monitor = self.config.early_stopping_monitor
        mode = self.config.early_stopping_mode
        min_delta = self.config.early_stopping_min_delta
        patience = self.config.early_stopping_patience
        
        current_value = val_results.get(monitor, val_results["loss"])
        
        if mode == "min":
            improved = current_value < self.best_val_loss - min_delta
            if improved:
                self.best_val_loss = current_value
        else:  # mode == "max"
            improved = current_value > self.best_val_metric + min_delta
            if improved:
                self.best_val_metric = current_value
        
        if improved:
            self.early_stop_counter = 0
            return False
        else:
            self.early_stop_counter += 1
            if self.early_stop_counter >= patience:
                print(f"\n  [Early Stopping] No improvement in '{monitor}' for {patience} epochs. Stopping training.")
                return True
            return False

    def _save_top_k_checkpoint(self, metric: float, epoch: int):
        """Save checkpoint and maintain top-k best models."""
        filename = f"epoch_{epoch:03d}_metric_{metric:.4f}.pth"
        path = os.path.join(self.config.checkpoint_dir, filename)
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config.__dict__,
            "metric": metric,
            "history": self.history,
        }, path)
        
        self.top_k_checkpoints.append((metric, filename))
        self.top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Keep only top-k
        while len(self.top_k_checkpoints) > self.config.save_top_k:
            _, old_file = self.top_k_checkpoints.pop()
            old_path = os.path.join(self.config.checkpoint_dir, old_file)
            if os.path.exists(old_path):
                os.remove(old_path)
        
        # Also save as best_model.pth if this is the best
        if self.top_k_checkpoints[0][1] == filename:
            best_path = os.path.join(self.config.checkpoint_dir, "best_model.pth")
            if os.path.exists(best_path):
                os.remove(best_path)
            # Copy instead of symlink for Windows compatibility
            torch.save(torch.load(path, map_location='cpu'), best_path)

    def _save_last_checkpoint(self):
        """Save the last epoch checkpoint."""
        if not self.config.save_last:
            return
        path = os.path.join(self.config.checkpoint_dir, "last_model.pth")
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "config": self.config.__dict__,
            "metric": self.history["val_loss"][-1] if self.history["val_loss"] else float('inf'),
            "history": self.history,
        }, path)

    def fit(self) -> Dict[str, list]:
        """
        Main training loop with learning rate scheduling, early stopping, and checkpointing.
        Returns history dictionary with all tracked metrics.
        """
        print(f"\n{'='*60}")
        print(f"  STARTING TRAINING: {self.config}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            train_results = self.train_epoch()
            val_results = self.validate()

            # Step scheduler
            if self.scheduler:
                if self.scheduler_is_plateau:
                    self.scheduler.step(val_results["loss"])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            self.history["train_loss"].append(train_results["loss"])
            self.history["val_loss"].append(val_results["loss"])
            self.history["learning_rates"].append(current_lr)
            self.history["epoch_times"].append(epoch_time)

            if self.metrics_fn:
                self.history["train_metrics"].append(
                    {k: v for k, v in train_results.items() if k != "loss"}
                )
                self.history["val_metrics"].append(
                    {k: v for k, v in val_results.items() if k != "loss"}
                )

            print(f"\n>> Epoch [{epoch+1}/{self.config.num_epochs}] "
                  f"| Train Loss: {train_results['loss']:.4f} "
                  f"| Val Loss: {val_results['loss']:.4f} "
                  f"| Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")

            # Determine monitoring metric for checkpointing/early stopping
            if self.metrics_fn:
                monitor_metric = val_results.get(self.config.early_stopping_monitor, val_results["loss"])
            else:
                monitor_metric = val_results["loss"]

            # Save top-k checkpoint
            self._save_top_k_checkpoint(monitor_metric, epoch)
            
            # Save last checkpoint
            self._save_last_checkpoint()

            # Check early stopping
            if self._check_early_stopping(val_results):
                break

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"  Best Val Metric: {self.top_k_checkpoints[0][0] if self.top_k_checkpoints else 'N/A':.4f}")
        print(f"  Checkpoints saved: {len(self.top_k_checkpoints)}")
        for metric, fname in self.top_k_checkpoints:
            print(f"    {fname} (metric: {metric:.4f})")
        print(f"{'='*60}\n")
        return self.history

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True, load_scheduler: bool = True) -> int:
        """
        Load model from checkpoint for resuming training or inference.
        Returns the epoch number to resume from.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if load_scheduler and self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.history = checkpoint.get("history", self.history)
        self.current_epoch = checkpoint.get("epoch", -1) + 1
        
        print(f"  Loaded checkpoint from epoch {self.current_epoch}")
        return self.current_epoch


def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Module 10: Factory function to create train/val DataLoaders with optimal settings.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for training (affects gradient estimate quality & GPU memory)
        num_workers: Number of subprocesses for data loading (0 = main process)
        pin_memory: Use pinned memory for faster CPU->GPU transfer
    
    Returns:
        (train_loader, val_loader) tuple
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def compare_optimizers(
    model_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizers: list[str] = ["adam", "adamw", "sgd"],
    lr: float = 1e-4,
    epochs: int = 3,
    device: str = "auto",
) -> Dict[str, Dict]:
    """
    Module 10: Utility to compare optimizer performance.
    Trains the same model architecture with different optimizers for a few epochs.
    """
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)

    for opt_name in optimizers:
        print(f"\n--- Testing {opt_name.upper()} ---")
        model = model_fn().to(device)
        config = TrainingConfig(
            batch_size=train_loader.batch_size,
            learning_rate=lr,
            optimizer_name=opt_name,
            num_epochs=epochs,
            device=device,
            mixed_precision=device.type == "cuda",
        )
        trainer = Trainer(model, train_loader, val_loader, criterion, config)
        history = trainer.fit()
        results[opt_name] = {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
            "history": history,
        }
    
    print("\n" + "="*60)
    print("  OPTIMIZER COMPARISON RESULTS")
    print("="*60)
    for opt, res in results.items():
        print(f"  {opt.upper():<8} | Train: {res['final_train_loss']:.4f} | "
              f"Val: {res['final_val_loss']:.4f} | Best Val: {res['best_val_loss']:.4f}")
    print("="*60)
    return results


# ==============================================================================
# HYPERPARAMETER TUNING UTILITIES
# ==============================================================================

import itertools
import random
from dataclasses import dataclass
from typing import Any

@dataclass
class HPConfig:
    """Single hyperparameter configuration for search."""
    learning_rate: float
    batch_size: int
    optimizer_name: str
    weight_decay: float
    scheduler_name: str
    num_epochs: int = 10
    
    def to_training_config(self, device: str = "auto", **kwargs) -> TrainingConfig:
        return TrainingConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            optimizer_name=self.optimizer_name,
            weight_decay=self.weight_decay,
            scheduler_name=self.scheduler_name,
            num_epochs=self.num_epochs,
            device=device,
            **kwargs
        )

def generate_hp_grid(search_space: Dict[str, List]) -> List[HPConfig]:
    """
    Generate grid of hyperparameter configurations from search space.
    
    Args:
        search_space: Dict mapping parameter names to lists of values
        
    Returns:
        List of HPConfig objects covering all combinations
    """
    keys = list(search_space.keys())
    values = list(search_space.values())
    combinations = list(itertools.product(*values))
    
    configs = []
    for combo in combinations:
        config_dict = dict(zip(keys, combo))
        configs.append(HPConfig(**config_dict))
    
    return configs

def sample_hp_random(search_space: Dict[str, List], n_samples: int, seed: int = 42) -> List[HPConfig]:
    """
    Random sampling from hyperparameter search space.
    
    Args:
        search_space: Dict mapping parameter names to lists of values
        n_samples: Number of random configurations to sample
        seed: Random seed for reproducibility
        
    Returns:
        List of HPConfig objects
    """
    random.seed(seed)
    configs = []
    for _ in range(n_samples):
        config_dict = {k: random.choice(v) for k, v in search_space.items()}
        configs.append(HPConfig(**config_dict))
    return configs

def run_hp_search(
    model_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    hp_configs: List[HPConfig],
    device: str = "auto",
    metrics_fn: Optional[Callable] = None,
    early_stopping_patience: int = 5,
) -> List[Dict]:
    """
    Run hyperparameter search over multiple configurations.
    
    Args:
        model_fn: Callable that returns a new model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        hp_configs: List of HPConfig objects to try
        device: Device to train on
        metrics_fn: Optional metrics function
        early_stopping_patience: Patience for early stopping
        
    Returns:
        List of result dictionaries with config and metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    results = []
    
    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SEARCH: {len(hp_configs)} configurations")
    print(f"{'='*70}\n")
    
    for i, hp_config in enumerate(hp_configs):
        print(f"\n[{i+1}/{len(hp_configs)}] Testing: {hp_config}")
        
        model = model_fn().to(device)
        config = hp_config.to_training_config(
            device=device,
            early_stopping_patience=early_stopping_patience,
            checkpoint_dir=f"hp_search/trial_{i}",
            log_interval=100,  # Less verbose for HP search
        )
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            metrics_fn=metrics_fn,
        )
        
        try:
            history = trainer.fit()
            
            best_val_loss = min(history["val_loss"]) if history["val_loss"] else float('inf')
            best_val_metric = max(
                [m.get("iou", m.get("f1", -l)) for m, l in zip(history.get("val_metrics", []), history["val_loss"])],
                default=-float('inf')
            )
            
            result = {
                "config": hp_config.__dict__,
                "best_val_loss": best_val_loss,
                "best_val_metric": best_val_metric,
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                "epochs_trained": len(history["train_loss"]),
                "history": history,
            }
            results.append(result)
            
            print(f"  Result: Val Loss={best_val_loss:.4f}, Val Metric={best_val_metric:.4f}, Epochs={result['epochs_trained']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "config": hp_config.__dict__,
                "error": str(e),
            })
    
    # Sort by best validation metric (higher is better for IoU/F1)
    results.sort(key=lambda x: x.get("best_val_metric", -float('inf')), reverse=True)
    
    print(f"\n{'='*70}")
    print(f"  HYPERPARAMETER SEARCH COMPLETE - TOP 5 RESULTS")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'LR':<10} {'BS':<5} {'Opt':<8} {'WD':<8} {'Sched':<12} {'Best Val Loss':<14} {'Best Metric':<12}")
    print("-" * 70)
    for rank, r in enumerate(results[:5], 1):
        c = r["config"]
        print(f"{rank:<5} {c['learning_rate']:<10.2e} {c['batch_size']:<5} {c['optimizer_name']:<8} "
              f"{c['weight_decay']:<8.2e} {c['scheduler_name']:<12} "
              f"{r.get('best_val_loss', float('inf')):<14.4f} {r.get('best_val_metric', -float('inf')):<12.4f}")
    print("="*70)
    
    return results


def get_gpu_info() -> Dict[str, Any]:
    """
    Module 10: Get GPU information for training optimization.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
    }
    
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        for i in range(info["device_count"]):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })
        info["current_device"] = torch.cuda.current_device()
        info["memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["memory_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    
    return info


def estimate_batch_size(model: nn.Module, input_shape: tuple, device: str = "cuda", 
                        target_memory_gb: float = 0.9, dtype: torch.dtype = torch.float32) -> int:
    """
    Module 10: Estimate maximum batch size that fits in GPU memory.
    
    Args:
        model: Model to test
        input_shape: Input tensor shape (C, H, W) for single sample
        device: Target device
        target_memory_gb: Fraction of GPU memory to use (0.9 = 90%)
        dtype: Data type for tensors
        
    Returns:
        Estimated maximum batch size
    """
    if not torch.cuda.is_available() or device == "cpu":
        return 32  # Default for CPU
    
    model = model.to(device)
    model.eval()
    
    # Get GPU memory
    total_memory = torch.cuda.get_device_properties(0).total_memory
    target_bytes = total_memory * target_memory_gb
    
    # Binary search for max batch size
    low, high = 1, 256
    best = 1
    
    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            x1 = torch.randn(mid, *input_shape, device=device, dtype=dtype)
            x2 = torch.randn(mid, *input_shape, device=device, dtype=dtype)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=True):
                    _ = model(x1, x2)
            
            allocated = torch.cuda.memory_allocated()
            if allocated < target_bytes:
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
        except Exception:
            high = mid - 1
    
    return max(1, best)


# ==============================================================================
# OVERFITTING ANALYSIS & MULTI-ARCHITECTURE TRAINING
# ==============================================================================

def analyze_overfitting(history: Dict[str, list], window: int = 5) -> Dict[str, Any]:
    """
    Analyze training history for signs of overfitting.
    
    Returns:
        Dictionary with overfitting metrics and recommendations
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    train_metrics = history.get("train_metrics", [])
    val_metrics = history.get("val_metrics", [])
    
    if len(train_loss) < 2 or len(val_loss) < 2:
        return {"status": "insufficient_data", "message": "Need at least 2 epochs"}
    
    # Gap between train and val loss (overfitting indicator)
    loss_gap = [v - t for t, v in zip(train_loss, val_loss)]
    avg_gap = np.mean(loss_gap[-window:]) if len(loss_gap) >= window else np.mean(loss_gap)
    
    # Trend analysis
    recent_train = train_loss[-window:] if len(train_loss) >= window else train_loss
    recent_val = val_loss[-window:] if len(val_loss) >= window else val_loss
    
    train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0] if len(recent_train) > 1 else 0
    val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0] if len(recent_val) > 1 else 0
    
    # Metric gap (if available)
    metric_gap = 0
    if train_metrics and val_metrics:
        train_iou = [m.get("iou", 0) for m in train_metrics[-window:]]
        val_iou = [m.get("iou", 0) for m in val_metrics[-window:]]
        if train_iou and val_iou:
            metric_gap = np.mean([t - v for t, v in zip(train_iou, val_iou)])
    
    # Determine status
    if avg_gap > 0.1 and val_trend > 0:
        status = "overfitting"
        message = "Validation loss increasing while train loss decreasing - model is overfitting"
    elif avg_gap > 0.05:
        status = "mild_overfitting"
        message = "Moderate gap between train and val loss"
    elif val_trend > 0.01:
        status = "val_increasing"
        message = "Validation loss trending upward"
    elif train_trend > 0:
        status = "underfitting"
        message = "Both train and val loss increasing - model not learning"
    else:
        status = "healthy"
        message = "Training appears healthy"
    
    return {
        "status": status,
        "message": message,
        "avg_loss_gap": avg_gap,
        "train_loss_trend": train_trend,
        "val_loss_trend": val_trend,
        "metric_gap": metric_gap,
        "recommendations": _get_overfitting_recommendations(status, avg_gap),
    }


def _get_overfitting_recommendations(status: str, gap: float) -> List[str]:
    """Get recommendations based on overfitting analysis."""
    recommendations = []
    
    if status in ["overfitting", "mild_overfitting"]:
        recommendations.extend([
            "Increase data augmentation",
            "Add dropout or increase dropout rate",
            "Add weight decay (L2 regularization)",
            "Reduce model capacity (fewer channels/layers)",
            "Use stronger early stopping",
            "Increase training data if possible",
        ])
    elif status == "val_increasing":
        recommendations.extend([
            "Reduce learning rate",
            "Increase early stopping patience",
            "Add learning rate scheduling (cosine annealing)",
        ])
    elif status == "underfitting":
        recommendations.extend([
            "Increase model capacity",
            "Train for more epochs",
            "Reduce regularization",
            "Increase learning rate",
        ])
    else:
        recommendations.append("Training is healthy - continue current setup")
    
    return recommendations


def train_multiple_architectures(
    model_configs: List[Dict],
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    base_config: TrainingConfig,
    metrics_fn: Optional[Callable] = None,
    device: str = "auto",
) -> List[Dict]:
    """
    Train multiple model architectures with the same training setup.
    
    Args:
        model_configs: List of dicts with keys: 'name', 'model_fn', 'model_kwargs'
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        base_config: Base TrainingConfig (will be copied for each model)
        metrics_fn: Metrics function
        device: Device to train on
        
    Returns:
        List of result dictionaries for each architecture
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    results = []
    
    print(f"\n{'='*70}")
    print(f"  MULTI-ARCHITECTURE TRAINING: {len(model_configs)} models")
    print(f"{'='*70}\n")
    
    for i, model_cfg in enumerate(model_configs):
        name = model_cfg["name"]
        model_fn = model_cfg["model_fn"]
        model_kwargs = model_cfg.get("model_kwargs", {})
        
        print(f"\n[{i+1}/{len(model_configs)}] Training: {name}")
        
        model = model_fn(**model_kwargs).to(device)
        
        # Create config copy with model-specific checkpoint dir
        config = TrainingConfig(**{**base_config.__dict__, "checkpoint_dir": f"{base_config.checkpoint_dir}/{name}"})
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
            metrics_fn=metrics_fn,
        )
        
        try:
            history = trainer.fit()
            
            # Analyze overfitting
            overfitting_analysis = analyze_overfitting(history)
            
            best_val_loss = min(history["val_loss"]) if history["val_loss"] else float('inf')
            best_val_metric = max(
                [m.get("iou", m.get("f1", -l)) for m, l in zip(history.get("val_metrics", []), history["val_loss"])],
                default=-float('inf')
            )
            
            result = {
                "name": name,
                "config": model_kwargs,
                "best_val_loss": best_val_loss,
                "best_val_metric": best_val_metric,
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
                "epochs_trained": len(history["train_loss"]),
                "overfitting": overfitting_analysis,
                "history": history,
            }
            results.append(result)
            
            print(f"  Result: Val Loss={best_val_loss:.4f}, Val Metric={best_val_metric:.4f}")
            print(f"  Overfitting: {overfitting_analysis['status']} - {overfitting_analysis['message']}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "name": name,
                "error": str(e),
            })
    
    # Sort by best validation metric
    results.sort(key=lambda x: x.get("best_val_metric", -float('inf')), reverse=True)
    
    print(f"\n{'='*70}")
    print(f"  MULTI-ARCHITECTURE RESULTS (Ranked by Val Metric)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Model':<30} {'Val Loss':<10} {'Val Metric':<12} {'Overfitting':<15} {'Epochs':<7}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        if "error" not in r:
            overfit_status = r["overfitting"]["status"]
            print(f"{rank:<5} {r['name']:<30} {r['best_val_loss']:<10.4f} {r['best_val_metric']:<12.4f} "
                  f"{overfit_status:<15} {r['epochs_trained']:<7}")
    print("="*70)
    
    return results


def plot_training_history(history: Dict[str, list], save_path: Optional[str] = None):
    """
    Plot training/validation loss and metrics.
    Requires matplotlib.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[0, 1].plot(epochs, history["learning_rates"], 'g-')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Metrics (if available)
    if history.get("train_metrics") and history.get("val_metrics"):
        metrics_keys = history["train_metrics"][0].keys() if history["train_metrics"] else []
        for key in metrics_keys:
            train_vals = [m.get(key, 0) for m in history["train_metrics"]]
            val_vals = [m.get(key, 0) for m in history["val_metrics"]]
            axes[1, 0].plot(epochs, train_vals, '--', label=f'Train {key}')
            axes[1, 0].plot(epochs, val_vals, '-', label=f'Val {key}')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Metrics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting gap
        if history["train_metrics"] and history["val_metrics"]:
            iou_gap = [t.get("iou", 0) - v.get("iou", 0) 
                       for t, v in zip(history["train_metrics"], history["val_metrics"])]
            axes[1, 1].plot(epochs, iou_gap, 'm-')
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Train IoU - Val IoU')
            axes[1, 1].set_title('Overfitting Gap (IoU)')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to: {save_path}")
    plt.close()


def compare_loss_functions(
    model_fn: Callable,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_functions: List[str] = ["bce", "dice", "focal", "bce_dice", "bce_focal"],
    lr: float = 1e-4,
    epochs: int = 5,
    device: str = "auto",
    metrics_fn: Optional[Callable] = None,
) -> Dict[str, Dict]:
    """
    Compare different loss functions on the same model/architecture.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    results = {}
    
    print(f"\n{'='*60}")
    print(f"  LOSS FUNCTION COMPARISON")
    print(f"{'='*60}\n")
    
    for loss_name in loss_functions:
        print(f"\n--- Testing {loss_name.upper()} ---")
        
        model = model_fn().to(device)
        criterion = get_loss_function(loss_name)
        
        config = TrainingConfig(
            batch_size=train_loader.batch_size,
            learning_rate=lr,
            optimizer_name="adamw",
            num_epochs=epochs,
            device=device,
            mixed_precision=device.type == "cuda",
            early_stopping_patience=10,
            checkpoint_dir=f"loss_comparison/{loss_name}",
        )
        
        trainer = Trainer(model, train_loader, val_loader, criterion, config, metrics_fn)
        history = trainer.fit()
        
        best_val_loss = min(history["val_loss"]) if history["val_loss"] else float('inf')
        best_val_metric = max(
            [m.get("iou", m.get("f1", -l)) for m, l in zip(history.get("val_metrics", []), history["val_loss"])],
            default=-float('inf')
        )
        
        results[loss_name] = {
            "best_val_loss": best_val_loss,
            "best_val_metric": best_val_metric,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "epochs_trained": len(history["train_loss"]),
            "history": history,
        }
    
    print(f"\n{'='*60}")
    print(f"  LOSS FUNCTION COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"{'Loss Function':<15} {'Val Loss':<10} {'Val Metric':<12} {'Epochs':<7}")
    print("-" * 60)
    for loss_name, res in sorted(results.items(), key=lambda x: x[1]["best_val_metric"], reverse=True):
        print(f"{loss_name:<15} {res['best_val_loss']:<10.4f} {res['best_val_metric']:<12.4f} {res['epochs_trained']:<7}")
    print("="*60)
    
    return results