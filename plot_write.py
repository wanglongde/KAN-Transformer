import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import glob
import numpy as np

def write_metrics_summary(save_dir):
    """
    Write relevant metrics from log files to a summary text file.
    """
    try:
        paths = glob.glob(os.path.join(save_dir, "*_log.tsv"))
        if not paths:
            print(f"No log files found in {save_dir}")
            return
        
        summary_file = os.path.join(save_dir, "metrics_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("MODEL TRAINING METRICS SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {pd.Timestamp.now()}\n")
            f.write(f"Total models: {len(paths)}\n\n")
            
            all_models_data = []
            
            for p in paths:
                try:
                    df = pd.read_csv(p, sep="\t", comment='#')
                    if len(df) == 0:
                        continue
                    
                    model_name = os.path.basename(p).replace("_log.tsv", "")
                    
                    # Get final metrics
                    final_loss = df['test_loss'].iloc[-1]
                    final_perplexity = df['test_perplexity'].iloc[-1]
                    total_steps_logged = len(df)
                    max_step = df['step'].max()  # Actual total training steps
                    final_training_time = df['training_time_sec'].iloc[-1]  # Total training time
                    
                    # Get best metrics
                    best_loss_idx = df['test_loss'].idxmin()
                    best_loss = df['test_loss'].iloc[best_loss_idx]
                    best_perplexity = df['test_perplexity'].iloc[best_loss_idx]
                    best_step = df['step'].iloc[best_loss_idx]  # Step where best occurred
                    best_training_time = df['training_time_sec'].iloc[best_loss_idx]  # Time to best
                    
                    initial_loss = df['test_loss'].iloc[0]
                    final_loss_val = df['test_loss'].iloc[-1]
                    
                    # Convergence threshold is test_loss <= 1
                    #min_loss_threshold = best_loss * 1.1
                    min_loss_threshold = 1
                    convergence_data = df[df['test_loss'] <= min_loss_threshold]
                    
                    if len(convergence_data) > 0:
                        convergence_step = convergence_data['step'].iloc[0]  # Use actual step number
                        convergence_time = df[df['step'] == convergence_step]['training_time_sec'].iloc[0] if convergence_step in df['step'].values else final_training_time
                        convergence_percentage = (convergence_step / max_step) * 100
                    else:
                        convergence_step = max_step
                        convergence_time = final_training_time
                        convergence_percentage = 100
                    
                    # Calculate training stability (using actual step spacing)
                    if len(df) > 5:
                        last_points = df.tail(5)
                        final_loss_std = last_points['test_loss'].std()
                        # Calculate approximate steps between evaluations
                        avg_step_gap = (df['step'].iloc[-1] - df['step'].iloc[0]) / (len(df) - 1) if len(df) > 1 else 0
                    else:
                        final_loss_std = 0
                        avg_step_gap = 0
                    
                    # Calculate improvement
                    improvement_pct = ((initial_loss - final_loss_val) / initial_loss * 100) if initial_loss > 0 else 0
                    
                    # Calculate efficiency metrics (using steps and time instead of FLOPs)
                    steps_to_converge = convergence_step
                    time_to_converge = convergence_time
                    
                    model_data = {
                        'name': model_name,
                        'final_loss': final_loss,
                        'final_ppl': final_perplexity,
                        'best_loss': best_loss,
                        'best_ppl': best_perplexity,
                        'best_step': best_step,
                        'total_steps_logged': total_steps_logged,
                        'max_step': max_step,
                        'convergence_step': convergence_step,
                        'convergence_percentage': convergence_percentage,
                        'loss_std': final_loss_std,
                        'initial_loss': initial_loss,
                        'improvement_pct': improvement_pct,
                        'avg_step_gap': avg_step_gap,
                        'steps_to_converge': steps_to_converge,
                        'total_training_time': final_training_time,
                        'time_to_converge': time_to_converge,
                        'time_to_best': best_training_time
                    }
                    all_models_data.append(model_data)
                    
                except Exception as e:
                    print(f"Error processing {p}: {e}")
                    continue
            
            # Sort by final perplexity 
            all_models_data.sort(key=lambda x: x['final_ppl'])
            
            f.write("DETAILED MODEL METRICS (sorted by final perplexity)\n")
            f.write("=" * 50 + "\n")
            
            for i, model in enumerate(all_models_data):
                f.write(f"\n{i+1}. {model['name']}\n")
                f.write(f"   Final Test Loss:     {model['final_loss']:.6f}\n")
                f.write(f"   Final Perplexity:    {model['final_ppl']:.6f}\n")
                f.write(f"   Best Test Loss:      {model['best_loss']:.6f} (at step {model['best_step']})\n")
                f.write(f"   Best Perplexity:     {model['best_ppl']:.6f}\n")
                f.write(f"   Total Steps:         {model['max_step']} (actual training)\n")
                f.write(f"   Total Time:          {model['total_training_time']:.1f}s\n")
                f.write(f"   Time to Best:        {model['time_to_best']:.1f}s\n")
                f.write(f"   Logged Points:       {model['total_steps_logged']} evaluations\n")
                f.write(f"   Convergence:         step {model['convergence_step']} ({model['convergence_percentage']:.1f}% of training)\n")
                f.write(f"   Steps to Converge:   {model['steps_to_converge']}\n")
                f.write(f"   Time to Converge:    {model['time_to_converge']:.1f}s\n")
                f.write(f"   Loss Stability:      {model['loss_std']:.6f} (std in last 5 evals)\n")
                f.write(f"   Improvement:         {model['improvement_pct']:.1f}% (from initial)\n")
                f.write(f"   Eval Frequency:      ~{model['avg_step_gap']:.0f} steps between evals\n")
            
            f.write("\n\nCOMPARATIVE ANALYSIS\n")
            f.write("=" * 50 + "\n")
            
            if len(all_models_data) > 1:
                best_model = all_models_data[0]
                worst_model = all_models_data[-1]
                
                f.write(f"Best Model: {best_model['name']}\n")
                f.write(f"Worst Model: {worst_model['name']}\n\n")
                
                f.write("Performance Comparison:\n")
                ppl_improvement = (worst_model['final_ppl'] - best_model['final_ppl']) / worst_model['final_ppl'] * 100
                time_ratio = best_model['total_training_time'] / worst_model['total_training_time']
                convergence_speed_ratio = worst_model['convergence_step'] / best_model['convergence_step']
                convergence_time_ratio = worst_model['time_to_converge'] / best_model['time_to_converge']
                
                f.write(f"  Perplexity Improvement: {ppl_improvement:.2f}%\n")
                f.write(f"  Training Time Ratio (best/worst): {time_ratio:.3f}\n")
                f.write(f"  Time Advantage: {1/time_ratio:.1f}x faster\n")
                f.write(f"  Convergence Speed: {convergence_speed_ratio:.1f}x faster in steps\n")
                f.write(f"  Convergence Time: {convergence_time_ratio:.1f}x faster in wall-clock time\n")
            
            f.write("\n\nKEY INSIGHTS\n")
            f.write("=" * 50 + "\n")
            
            # Group by model type for analysis
            mlp_models = [m for m in all_models_data if m['name'].startswith('mlp')]
            kan_models = [m for m in all_models_data if m['name'].startswith('kan')]
            
            if mlp_models and kan_models:
                avg_mlp_ppl = np.mean([m['final_ppl'] for m in mlp_models])
                avg_kan_ppl = np.mean([m['final_ppl'] for m in kan_models])
                avg_mlp_time = np.mean([m['total_training_time'] for m in mlp_models])
                avg_kan_time = np.mean([m['total_training_time'] for m in kan_models])
                avg_mlp_conv = np.mean([m['convergence_step'] for m in mlp_models])
                avg_kan_conv = np.mean([m['convergence_step'] for m in kan_models])
                avg_mlp_std = np.mean([m['loss_std'] for m in mlp_models])
                avg_kan_std = np.mean([m['loss_std'] for m in kan_models])
                avg_mlp_time_conv = np.mean([m['time_to_converge'] for m in mlp_models])
                avg_kan_time_conv = np.mean([m['time_to_converge'] for m in kan_models])
                
                f.write(f"MLP Models (n={len(mlp_models)}):\n")
                f.write(f"  Average Perplexity: {avg_mlp_ppl:.6f}\n")
                f.write(f"  Average Time:       {avg_mlp_time:.1f}s\n")
                f.write(f"  Avg Convergence:    {avg_mlp_conv:.0f} steps\n")
                f.write(f"  Avg Time to Conv:   {avg_mlp_time_conv:.1f}s\n")
                f.write(f"  Avg Loss Stability: {avg_mlp_std:.6f}\n\n")
                
                f.write(f"KAN Models (n={len(kan_models)}):\n")
                f.write(f"  Average Perplexity: {avg_kan_ppl:.6f}\n")
                f.write(f"  Average Time:       {avg_kan_time:.1f}s\n")
                f.write(f"  Avg Convergence:    {avg_kan_conv:.0f} steps\n")
                f.write(f"  Avg Time to Conv:   {avg_kan_time_conv:.1f}s\n")
                f.write(f"  Avg Loss Stability: {avg_kan_std:.6f}\n\n")
                
                if avg_kan_ppl < avg_mlp_ppl:
                    f.write("KAN models outperform MLP models in final perplexity.\n")
                else:
                    f.write("MLP models outperform KAN models in final perplexity.\n")
                    
                time_ratio = avg_mlp_time / avg_kan_time
                f.write(f"Training time ratio (MLP/KAN): {time_ratio:.3f}\n")
                
                conv_ratio = avg_mlp_conv / avg_kan_conv
                f.write(f"Convergence speed ratio (MLP/KAN): {conv_ratio:.2f}\n")
                
                time_conv_ratio = avg_mlp_time_conv / avg_kan_time_conv
                f.write(f"Time to convergence ratio (MLP/KAN): {time_conv_ratio:.2f}\n")
                
                stability_ratio = avg_mlp_std / avg_kan_std
                f.write(f"Stability ratio (MLP/KAN): {stability_ratio:.2f}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("=" * 50 + "\n")
            best_model = all_models_data[0]
            f.write(f"1. Best performing model: {best_model['name']}\n")
            f.write(f"2. Consider this model for production use\n")
            
            # Find most efficient model (best performance per training time)
            if len(all_models_data) > 1:
                efficiency_scores = [(m['final_ppl'] / m['total_training_time'], m) for m in all_models_data]
                most_efficient = min(efficiency_scores, key=lambda x: x[0])[1]
                f.write(f"3. Most time-efficient model: {most_efficient['name']} (PPL/sec: {most_efficient['final_ppl'] / most_efficient['total_training_time']:.2e})\n")
            
            if len(kan_models) > 0:
                best_kan = min(kan_models, key=lambda x: x['final_ppl'])
                f.write(f"4. Best KAN model: {best_kan['name']} (PPL: {best_kan['final_ppl']:.6f})\n")
            
            # Find fastest converging model
            fastest_conv = min(all_models_data, key=lambda x: x['convergence_step'])
            f.write(f"5. Fastest converging model: {fastest_conv['name']} (converged at step {fastest_conv['convergence_step']})\n")
            
            # Find fastest in wall-clock time
            fastest_time = min(all_models_data, key=lambda x: x['time_to_converge'])
            f.write(f"6. Fastest wall-clock convergence: {fastest_time['name']} (converged in {fastest_time['time_to_converge']:.1f}s)\n")
            
            stable_models = [m for m in all_models_data if m['loss_std'] < 0.001]
            if stable_models:
                f.write(f"7. {len(stable_models)} models show excellent training stability (std < 0.001)\n")
            
            early_convergers = [m for m in all_models_data if m['convergence_percentage'] < 50]
            if early_convergers:
                f.write(f"8. {len(early_convergers)} models converged early (before 50% of training)\n")
        
        print(f"[metrics] Summary written to: {summary_file}")
        
        print(f"\n{'='*40}")
        print("QUICK SUMMARY")
        print(f"{'='*40}")
        for model in all_models_data[:3]:  # Top 3 models
            print(f"{model['name']:30} -> PPL: {model['final_ppl']:.6f}, Steps: {model['max_step']}, Time: {model['total_training_time']:.1f}s")
            
    except Exception as e:
        print(f"[metrics error] {e}")
        import traceback
        traceback.print_exc()


def plot_logs_clean(save_dir, start_steps=[400], end_steps=[None]):
    """
    Create plots with labeling showing KAN parameters (width factor and grid size).
    """
    try:
        # Validate inputs
        if len(start_steps) != len(end_steps):
            raise ValueError("start_steps and end_steps must have same length")
        
        paths = glob.glob(os.path.join(save_dir, "*_log.tsv"))
        if not paths:
            print(f"No log files found in {save_dir}")
            return
        
        print(f"Found {len(paths)} log files to plot")

        def extract_params_from_filename(filename):
            """
            Extract key parameters from filename.
            """
            import re
            params = {}
            basename = os.path.basename(filename).replace("_log.tsv", "")
            
            params['is_mlp'] = 'mlp' in basename
            params['model_type'] = 'MLP' if params['is_mlp'] else 'KAN'
            
            # Extract d_model and n_layers for plot title
            d_match = re.search(r'_d(\d+)', basename)
            l_match = re.search(r'_l(\d+)', basename)
            
            params['d_model'] = int(d_match.group(1)) if d_match else 0
            params['n_layers'] = int(l_match.group(1)) if l_match else 0
            
            if params['is_mlp']:
                params['grid'] = None
                params['width_factor'] = None
                params['plot_label'] = "MLP"
                params['legend_label'] = "MLP (baseline)"
                # Sorting key: MLP first, then d_model, then n_layers
                params['sort_key'] = (0, params['d_model'], params['n_layers'], 0, 0)
                # Unique identifier for color mapping
                params['unique_label'] = f"MLP-d{params['d_model']}-L{params['n_layers']}"
            else:
                # KAN specific parameters
                g_match = re.search(r'_g(\d+)', basename)
                wf_match = re.search(r'_wf([\d\.]+)', basename)
                
                params['grid'] = int(g_match.group(1)) if g_match else 0
                params['width_factor'] = float(wf_match.group(1)) if wf_match else 0
                
                # Format width factor for display (remove trailing zeros)
                wf_str = f"{params['width_factor']:.3f}".rstrip('0').rstrip('.')
                
                # Clean legend label
                params['plot_label'] = f"KAN-wf{wf_str}"
                params['legend_label'] = f"KAN (G={params['grid']}, WF={wf_str})"
                # Sorting key: KAN second, then grid, then width_factor, then d_model, then n_layers
                params['sort_key'] = (1, params['grid'], params['width_factor'], params['d_model'], params['n_layers'])
                # Unique identifier for color mapping (includes both grid AND width factor)
                params['unique_label'] = f"KAN-G{params['grid']}-WF{wf_str}"
            
            return params
        
        # Extract parameters for each file
        file_params = {}
        for p in paths:
            params = extract_params_from_filename(p)
            file_params[p] = params
        
        # Sort paths using the extracted sort_key
        paths = sorted(paths, key=lambda p: file_params[p]['sort_key'])
        
        # Determine comparison type for filename
        comparison_type = 'model'  # default
        
        # Check what we are comparing by looking at parameter variations
        if len(paths) > 1:
            # Collect unique values for each parameter type
            width_factors = set()
            grids = set()
            d_models = set()
            n_layers_set = set()
            
            for p in paths:
                params = file_params[p]
                if params['width_factor'] is not None:
                    width_factors.add(params['width_factor'])
                if params['grid'] is not None:
                    grids.add(params['grid'])
                d_models.add(params['d_model'])
                n_layers_set.add(params['n_layers'])
            
            # Determine primary comparison type
            if len(width_factors) > 1 and len(paths) <= 10:
                comparison_type = 'width'
            elif len(grids) > 1 and len(width_factors) == 1:
                comparison_type = 'grid'
            elif len(d_models) > 1 and len(n_layers_set) == 1:
                comparison_type = 'd_model'
            elif len(n_layers_set) > 1 and len(d_models) == 1:
                comparison_type = 'layer'
            elif len(paths) == 2 and any(params['is_mlp'] for params in file_params.values()) and \
                 any(not params['is_mlp'] for params in file_params.values()):
                comparison_type = 'mlp_vs_kan'
        
        print(f"Detected comparison type: {comparison_type}")
        
        # Create color mapping
        unique_labels = []
        label_to_color = {}
        cmap = plt.cm.tab10
        
        for p in paths:
            unique_label = file_params[p]['unique_label']
            if unique_label not in unique_labels:
                unique_labels.append(unique_label)
        
        for i, label in enumerate(unique_labels):
            label_to_color[label] = cmap(i % 10)
        
        # Process each step range
        for range_idx, (start_step, end_step) in enumerate(zip(start_steps, end_steps)):
            print(f"\n=== Plotting range {range_idx+1}: steps {start_step} to {end_step if end_step else 'end'} ===")
            
            # PLOT 1: Loss vs Steps
            print(f"Creating Loss vs Steps plot with all {len(paths)} models")
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            
            all_test_losses = []
            plotted_labels = set()  # Track which legend labels we've plotted
            
            # Process all files for loss vs steps plot
            for p in paths:
                try:
                    df = pd.read_csv(p, sep="\t", comment='#')
                    if len(df) == 0:
                        continue
                    
                    params = file_params[p]
                    unique_label = params['unique_label']
                    legend_label = params['legend_label']
                    
                    # Filter steps
                    if end_step:
                        df_clean = df[(df['step'] >= start_step) & (df['step'] <= end_step)]
                    else:
                        df_clean = df[df['step'] >= start_step]
                    
                    if len(df_clean) > 0:
                        all_test_losses.extend(df_clean['test_loss'].values)
                        color = label_to_color[unique_label]
                        
                        # Use different line styles for MLP vs KAN
                        linestyle = '-' if params['is_mlp'] else '--'
                        linewidth = 2.5 if params['is_mlp'] else 2
                        
                        # Only add to legend once per unique legend label
                        if legend_label not in plotted_labels:
                            ax1.plot(df_clean['step'], df_clean['test_loss'], 
                                    label=legend_label, 
                                    color=color, 
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                    alpha=0.8)
                            plotted_labels.add(legend_label)
                        else:
                            ax1.plot(df_clean['step'], df_clean['test_loss'], 
                                    color=color, 
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                    alpha=0.8)
                        
                except Exception as e:
                    print(f"Error loading {p}: {e}")
                    continue
            
            if all_test_losses:
                loss_ymin = max(0, min(all_test_losses) * 0.95)
                loss_ymax = max(all_test_losses) * 1.05
                
                # Get model parameters for title (from first model)
                first_params = file_params[paths[0]]
                title_params = f"d={first_params['d_model']}, L={first_params['n_layers']}"
                
                ax1.set_xlabel('Training Steps')
                ax1.set_ylabel('Test Loss')
                ax1.set_title(f'Test Loss vs Training Steps\n{title_params} (Steps {start_step} to {end_step if end_step else "end"})')
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(loss_ymin, loss_ymax)
                
                plt.tight_layout()
                # Auto-detect comparison type for filename
                filename1 = f"{comparison_type}_loss_steps{start_step}to{end_step if end_step else 'end'}.png"
                outp1 = os.path.join(save_dir, filename1)
                plt.savefig(outp1, dpi=150, bbox_inches='tight')
                print(f"[plot] saved loss plot: {outp1}")
                plt.close(fig1)
            
            # PLOT 2: Perplexity vs Steps
            print(f"Creating Perplexity vs Steps plot with all {len(paths)} models")
            
            fig2, ax2 = plt.subplots(figsize=(10, 7))
            
            all_test_perplexities = []
            plotted_labels = set()
            
            for p in paths:
                try:
                    df = pd.read_csv(p, sep="\t", comment='#')
                    if len(df) == 0:
                        continue
                    
                    params = file_params[p]
                    unique_label = params['unique_label']
                    legend_label = params['legend_label']
                    
                    # Filter steps
                    if end_step:
                        df_clean = df[(df['step'] >= start_step) & (df['step'] <= end_step)]
                    else:
                        df_clean = df[df['step'] >= start_step]
                    
                    if len(df_clean) > 0:
                        all_test_perplexities.extend(df_clean['test_perplexity'].values)
                        color = label_to_color[unique_label]
                        
                        # Use different line styles for MLP vs KAN
                        linestyle = '-' if params['is_mlp'] else '--'
                        linewidth = 2.5 if params['is_mlp'] else 2
                        
                        # Only add to legend once per unique legend label
                        if legend_label not in plotted_labels:
                            ax2.plot(df_clean['step'], df_clean['test_perplexity'], 
                                    label=legend_label, 
                                    color=color, 
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                    alpha=0.8)
                            plotted_labels.add(legend_label)
                        else:
                            ax2.plot(df_clean['step'], df_clean['test_perplexity'], 
                                    color=color, 
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                    alpha=0.8)
                        
                except Exception as e:
                    print(f"Error loading {p}: {e}")
                    continue
            
            if all_test_perplexities:
                ppl_ymin = 1
                ppl_ymax = max(all_test_perplexities) * 1.02
                
                # Get model parameters for title (from first model)
                first_params = file_params[paths[0]]
                title_params = f"d={first_params['d_model']}, L={first_params['n_layers']}"
                
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('Test Perplexity')
                ax2.set_title(f'Test Perplexity vs Training Steps\n{title_params} (Steps {start_step} to {end_step if end_step else "end"})')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(ppl_ymin, ppl_ymax)
                
                plt.tight_layout()
                # Auto-detect comparison type for filename
                filename2 = f"{comparison_type}_ppl_steps{start_step}to{end_step if end_step else 'end'}.png"
                outp2 = os.path.join(save_dir, filename2)
                plt.savefig(outp2, dpi=150, bbox_inches='tight')
                print(f"[plot] saved perplexity vs steps plot: {outp2}")
                plt.close(fig2)
        
        print(f"\n[plot] All plots saved to: {save_dir}")
        print(f"[plot] Comparison type detected: {comparison_type}")
        print(f"[plot] PNG files named as: {comparison_type}_[loss/ppl]_stepsXtoY.png")
        
    except Exception as e:
        print(f"[plot error] {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Plot existing training results")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing log files")
    parser.add_argument("--start_steps", type=int, nargs='+', default=[400], help="First steps to plot (multiple allowed)")
    parser.add_argument("--end_steps", type=int, nargs='*', default=[], help="Last steps to plot (optional, use 0 for None)")
    args = parser.parse_args()
    
    # Handle empty end_steps
    if not args.end_steps:
        end_steps = [None] * len(args.start_steps)
    else:
        # Convert 0 to None for end_steps
        end_steps = [None if end == 0 else end for end in args.end_steps]
    
    plot_logs_clean(args.save_dir, args.start_steps, end_steps)
    write_metrics_summary(args.save_dir)


if __name__ == "__main__":
    main()