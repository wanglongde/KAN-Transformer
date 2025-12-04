import os
import time
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformer import MiniTransformerLM, count_params
import numpy as np
import sys
from contextlib import redirect_stdout, redirect_stderr
import pandas as pd

class TeeLogger:
    """Logger that writes to both console and file"""
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
        self.file.flush()
        
    def flush(self):
        self.console.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

        
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across Python, PyTorch, and CUDA.
    """
    random.seed(seed)
    torch.manual_seed(seed)


class CharDataset(Dataset):
    """
    Dataset for character-level language modeling.
    """
    def __init__(self, data_tensor, block_size):
        self.data = data_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


def ensure_data(path, download_flag):
    """
    Ensure the dataset file exists, and download if requested and not present.
    """
    if os.path.exists(path):
        return
    if not download_flag:
        raise FileNotFoundError(f"{path} not found. Use --download to fetch it.")
    
    import urllib.request
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading Tiny Shakespeare to {path} ...")
    urllib.request.urlretrieve(url, path)
    print("Done.")


def build_loaders(txt_path, block_size, batch_size):
    """
    Build data loaders for training, validation, and test with proper split.
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    n = len(data)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    train_loader = DataLoader(
        CharDataset(train_data, block_size),
        batch_size=batch_size, 
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        CharDataset(val_data, block_size),
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )
    test_loader = DataLoader(
        CharDataset(test_data, block_size),
        batch_size=batch_size, 
        shuffle=False,
        drop_last=True
    )
    
    return len(vocab), train_loader, val_loader, test_loader


def evaluate(model, loader, device="cpu"):
    """
    Evaluate model on a dataset and compute average loss and perplexity.
    """
    model.eval()
    total_loss = 0.0
    tokens = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            
            total_loss += loss.item() * x.numel()
            tokens += x.numel()
    
    avg_loss = total_loss / tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    model.train()
    return avg_loss, perplexity


def train_one(model, train_loader, val_loader, test_loader, epochs, lr, eval_every, device, label, save_dir, vocab_size, batch_size, block_size):
    """
    Train a single model with comprehensive logging.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up comprehensive logging for this model
    log_file = os.path.join(save_dir, f"{label}_full_log.txt")
    model_logger = TeeLogger(log_file)
    
    # Redirect stdout for this model's training
    old_stdout = sys.stdout
    sys.stdout = model_logger
    
    try:
        # Training log file (TSV for plotting)
        # REMOVED cumulative_flops column
        tsv_logp = os.path.join(save_dir, f"{label}_log.tsv")
        with open(tsv_logp, "w", encoding="utf-8") as f:
            f.write("step\ttrain_loss\tval_loss\ttest_loss\tval_perplexity\ttest_perplexity\ttraining_time_sec\n")  # Added training_time_sec
        
        print(f"=== STARTING TRAINING FOR {label} ===")
        print(f"Timestamp: {pd.Timestamp.now()}")
        
        # Record parameter count for analysis
        param_count = count_params(model)
        print(f"Model parameters: {param_count / 1e3:.3f}k")
        print(f"Training config: epochs={epochs}, lr={lr}, eval_every={eval_every}")
        print(f"Data config: batch_size={batch_size}, block_size={block_size}, vocab_size={vocab_size}")
        print(f"Device: {device}")
        print("-" * 60)

        total_start_time = time.time()  # Start timer for total training time
        
        model.to(device)
        
        if hasattr(torch, 'compile') and device == 'cuda':
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        
        # Track best model for final reporting
        best_val_loss = float("inf")
        best_val_step = 0
        best_test_loss = float("inf")
        
        step = 0
        total_batches = len(train_loader)

        for ep in range(1, epochs + 1):
            epoch_start_time = time.time()
            epoch_losses = []
            
            print(f"\n--- Epoch {ep}/{epochs} ---")
            pbar = tqdm(train_loader, desc=f"[{label}] epoch {ep}", ncols=80, file=sys.stdout)
            
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                
                opt.zero_grad(set_to_none=True)
                _, loss = model(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                
                step += 1
                epoch_losses.append(loss.item())
                
                # Evaluate and log
                if step % eval_every == 0:
                    val_loss, val_perplexity = evaluate(model, val_loader, device)
                    test_loss, test_perplexity = evaluate(model, test_loader, device)
                    
                    # Track best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_val_step = step
                        best_test_loss = test_loss
                        torch.save(model.state_dict(), os.path.join(save_dir, f"{label}_best.pt"))
                        print(f"\n*** NEW BEST: val_loss={val_loss:.6f} at step {step} ***")
                    
                    # Progress bar
                    pbar.set_postfix({
                        'loss': f'{loss.item():.3f}',
                        'val': f'{val_loss:.3f}',
                        'test': f'{test_loss:.3f}',
                        'best': f'{best_val_loss:.3f}'
                    })
                    
                    # Log to TSV file (REMOVED cumulative_flops, added training_time_sec)
                    current_time = time.time() - total_start_time
                    with open(tsv_logp, "a", encoding="utf-8") as f:
                        f.write(f"{step}\t{loss.item():.6f}\t{val_loss:.6f}\t{test_loss:.6f}\t{val_perplexity:.6f}\t{test_perplexity:.6f}\t{current_time:.2f}\n")
            
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = np.mean(epoch_losses)
            
            # End-of-epoch evaluation
            train_loss, train_ppl = evaluate(model, train_loader, device)
            val_loss, val_ppl = evaluate(model, val_loader, device)
            test_loss, test_ppl = evaluate(model, test_loader, device)
            
            print(f"\nEpoch {ep} Summary:")
            print(f"  Time: {epoch_time:.1f}s")
            print(f"  Steps: {step}")
            print(f"  Avg train loss: {avg_epoch_loss:.4f}")
            print(f"  Final train loss: {train_loss:.4f} (ppl: {train_ppl:.2f})")
            print(f"  Val loss: {val_loss:.4f} (ppl: {val_ppl:.2f})")
            print(f"  Test loss: {test_loss:.4f} (ppl: {test_ppl:.2f})")
            print(f"  Best val so far: {best_val_loss:.4f} at step {best_val_step}")

        total_time = time.time() - total_start_time
        
        # Load best model for final evaluation
        best_model_path = os.path.join(save_dir, f"{label}_best.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path))
            print(f"\nLoaded best model from step {best_val_step}")
        
        final_train, final_train_ppl = evaluate(model, train_loader, device)
        final_val, final_val_ppl = evaluate(model, val_loader, device)
        final_test, final_test_ppl = evaluate(model, test_loader, device)
        
        print(f"\n" + "="*60)
        print(f"FINAL RESULTS - {label}")
        print("="*60)
        print(f"Best validation loss: {best_val_loss:.6f} (at step {best_val_step})")
        print(f"Corresponding test loss: {best_test_loss:.6f}")
        print(f"Final train loss: {final_train:.6f} (perplexity: {final_train_ppl:.6f})")
        print(f"Final val loss: {final_val:.6f} (perplexity: {final_val_ppl:.6f})")
        print(f"Final test loss: {final_test:.6f} (perplexity: {final_test_ppl:.6f})")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Total steps: {step}")
        print(f"Total parameters: {param_count} ({param_count/1e3:.3f}k)")
        print("="*60)
        
        # Final log entry 
        with open(tsv_logp, "a", encoding="utf-8") as f:
            f.write(f"# Training completed at step {step}\n")
            f.write(f"# Best validation loss: {best_val_loss:.6f} at step {best_val_step}\n")
            f.write(f"# Final train loss: {final_train:.6f} (perplexity: {final_train_ppl:.6f})\n")
            f.write(f"# Final val loss: {final_val:.6f} (perplexity: {final_val_ppl:.6f})\n")
            f.write(f"# Final test loss: {final_test:.6f} (perplexity: {final_test_ppl:.6f})\n")
            f.write(f"# Total training time: {total_time:.1f}s\n")
            f.write(f"# Total parameters: {param_count}\n")

        return final_train, final_val, final_test, total_time, step, best_val_loss, param_count

    finally:
        # Restore stdout and close logger
        sys.stdout = old_stdout
        model_logger.close()
        print(f"[{label}] Training completed. Full log saved to: {log_file}")


def main():
    """
    Main training script with comprehensive logging.
    """
    ap = argparse.ArgumentParser(description="Train and compare MLP vs KAN transformer models")
    ap.add_argument("--data", type=str, default="tiny_shakespeare.txt", 
                   help="Path to dataset file")
    ap.add_argument("--download", action="store_true",
                   help="Download dataset if not present")
    ap.add_argument("--block_size", type=int, default=16,
                   help="Sequence length (context window)")
    ap.add_argument("--batch_size", type=int, default=2048,
                   help="Batch size for training")
    ap.add_argument("--d_model", type=str, default="32",
                   help="Model dimension (single value or comma-separated list)")
    ap.add_argument("--n_heads", type=int, default=4,
                   help="Number of attention heads")
    ap.add_argument("--n_layers", type=str, default="3",
                   help="Number of transformer layers (single value or comma-separated list)")
    ap.add_argument("--d_ff_mult", type=int, default=4,
                   help="Feed-forward dimension multiplier (d_ff = d_model * d_ff_mult)")
    ap.add_argument("--kan_grid", type=str, default="3",
                   help="Grid size for KAN spline approximation (single value or comma-separated list)")
    ap.add_argument("--kan_width_factors", type=str, default="0.125",
                   help="Comma-separated list of KAN width factors to test")
    ap.add_argument("--epochs", type=int, default=3,  # Reduced from 5 to 3
                   help="Number of training epochs")
    ap.add_argument("--lr", type=float, default=3e-4,
                   help="Learning rate")
    ap.add_argument("--eval_every", type=int, default=200,  # Less frequent evaluation
                   help="Evaluate model every N steps")
    ap.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    ap.add_argument("--save_dir", type=str, default="runs",
                   help="Directory to save logs and checkpoints")
    ap.add_argument("--skip_mlp", action="store_true", 
                   help="Skip MLP baseline training")
    args = ap.parse_args()

    # Set up main experiment logging
    os.makedirs(args.save_dir, exist_ok=True)
    experiment_log_file = os.path.join(args.save_dir, "experiment_log.txt")
    experiment_logger = TeeLogger(experiment_log_file)
    sys.stdout = experiment_logger
    
    try:
        print("="*80)
        print("TRANSFORMER EXPERIMENT - MLP vs KAN COMPARISON")
        print("="*80)
        print(f"Experiment started: {pd.Timestamp.now()}")
        print(f"Save directory: {args.save_dir}")
        print(f"Command: {' '.join(sys.argv)}")
        print("-"*80)
        
        set_seed(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ensure_data(args.data, args.download)
        
        vocab_size, train_loader, val_loader, test_loader = build_loaders(
            args.data, args.block_size, args.batch_size
        )

        def parse_list(arg_value, dtype=float):
            if isinstance(arg_value, str):
                return [dtype(x.strip()) for x in arg_value.split(",")]
            return [arg_value]
        
        n_layers_list = parse_list(args.n_layers, int)
        d_model_list = parse_list(args.d_model, int) 
        kan_grid_list = parse_list(args.kan_grid, int)
        kan_width_factors_list = parse_list(args.kan_width_factors, float)
        
        print(f"Device: {device} | Vocab size: {vocab_size}")
        print(f"Testing configurations:")
        print(f"  d_model: {d_model_list}")
        print(f"  n_layers: {n_layers_list}")
        print(f"  kan_grid: {kan_grid_list}")
        print(f"  kan_width_factors: {kan_width_factors_list}")
        print(f"Architecture: block_size={args.block_size}, batch_size={args.batch_size}, d_ff_mult={args.d_ff_mult}")
        print(f"Training: epochs={args.epochs}, lr={args.lr}, eval_every={args.eval_every}")
        print("-"*80)

        results = []
        total_mlp = 0 if args.skip_mlp else len(d_model_list) * len(n_layers_list)
        total_kan = len(d_model_list) * len(n_layers_list) * len(kan_grid_list) * len(kan_width_factors_list)
        total_models = total_mlp + total_kan
        current_model = 0
        
        print(f"Total models to train: {total_models}")

        for d_model in d_model_list:
            for n_layers in n_layers_list:
                d_ff = args.d_ff_mult * d_model
                
                print(f"\n{'='*80}")
                print(f"CONFIGURATION: d_model={d_model}, n_layers={n_layers}, d_ff={d_ff}")
                print(f"{'='*80}")

                if not args.skip_mlp:
                    current_model += 1
                    print(f"\n[{current_model}/{total_models}] TRAINING MLP BASELINE")
                    baseline = MiniTransformerLM(
                        vocab_size=vocab_size,
                        d_model=d_model,
                        n_heads=args.n_heads,
                        n_layers=n_layers,
                        d_ff=d_ff,
                        block_size=args.block_size,
                        use_kan_ffn=False,
                    )
                    # Add model attributes for reference
                    baseline.d_model = d_model
                    baseline.n_layers = n_layers
                    baseline.n_heads = args.n_heads
                    baseline.d_ff = d_ff
                    
                    base_train, base_val, base_test, base_total_time, base_steps, base_best_val, base_params = train_one(
                        baseline, train_loader, val_loader, test_loader,
                        epochs=args.epochs, lr=args.lr, eval_every=args.eval_every,
                        device=device, label=f"mlp_d{d_model}_l{n_layers}", save_dir=args.save_dir,
                        vocab_size=vocab_size, batch_size=args.batch_size, block_size=args.block_size
                    )
                    results.append((f"mlp_d{d_model}_l{n_layers}", base_test, base_best_val, base_params, base_total_time))

                for kan_grid in kan_grid_list:
                    for width_factor in kan_width_factors_list:
                        current_model += 1
                        print(f"\n[{current_model}/{total_models}] TRAINING KAN: grid={kan_grid}, wf={width_factor}")
                        
                        kan_model = MiniTransformerLM(
                            vocab_size=vocab_size,
                            d_model=d_model,
                            n_heads=args.n_heads,
                            n_layers=n_layers,
                            d_ff=d_ff,
                            block_size=args.block_size,
                            use_kan_ffn=True,
                            kan_grid=kan_grid,
                            kan_width_factor=width_factor,
                        )
                        # Add model attributes for reference
                        kan_model.d_model = d_model
                        kan_model.n_layers = n_layers
                        kan_model.n_heads = args.n_heads
                        kan_model.d_ff = d_ff
                        
                        kan_train, kan_val, kan_test, kan_total_time, kan_steps, kan_best_val, kan_params = train_one(
                            kan_model, train_loader, val_loader, test_loader,
                            epochs=args.epochs, lr=args.lr, eval_every=args.eval_every,
                            device=device, label=f"kan_d{d_model}_l{n_layers}_g{kan_grid}_wf{width_factor}", save_dir=args.save_dir,
                            vocab_size=vocab_size, batch_size=args.batch_size, block_size=args.block_size
                        )
                        results.append((f"kan_d{d_model}_l{n_layers}_g{kan_grid}_wf{width_factor}", kan_test, kan_best_val, kan_params, kan_total_time))

        print(f"\n{'='*80}")
        print("EXPERIMENT COMPLETE - SUMMARY")
        print(f"{'='*80}")
        print(f"Experiment finished: {pd.Timestamp.now()}")
        
        # Sort by best validation loss
        results.sort(key=lambda x: x[2])
        
        print(f"\nFINAL MODEL RANKINGS (by best validation loss):")
        for i, (label, test_loss, best_val_loss, params, train_time) in enumerate(results):
            test_perplexity = torch.exp(torch.tensor(test_loss)).item()
            print(f"{i+1:2d}. {label:40} | val_loss: {best_val_loss:.6f} | test_loss: {test_loss:.6f} | test_ppl: {test_perplexity:.6f} | params: {params/1e6:.3f}M | time: {train_time:.1f}s")
        
        print(f"\nExperiment data saved to: {args.save_dir}")
        print("Files created:")
        print(f"  - experiment_log.txt (this file)")
        for label, _, _, _, _ in results:
            print(f"  - {label}_full_log.txt (detailed training log)")
            print(f"  - {label}_log.tsv (plotting data - steps vs performance)")
            print(f"  - {label}_best.pt (model weights)")
        
        print(f"\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")

    finally:
        # Restore stdout
        sys.stdout = sys.__stdout__
        experiment_logger.close()
        print(f"\nMain experiment log saved to: {experiment_log_file}")


if __name__ == "__main__":
    main()