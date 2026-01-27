#!/usr/bin/env python3
"""
Sub-Phase Level GPU Profiling Analysis for RLHF Training
Correlates fine-grained timing data with GPU metrics.

Usage:
    python analyze_subphase_metrics.py --gpu-csv <csv_file> --timing-log <jsonl_file> [--output-dir <dir>]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Operation colors for visualization
# Operation colors for visualization
OPERATION_COLORS = {
    'gen': '#e74c3c',              # Red - generation E2E
    'generate_sequences': '#ff7f7f',  # Light red - token generation
    'reward': '#f39c12',           # Orange - reward
    'old_log_prob': '#3498db',     # Blue - log prob
    'ref': '#9b59b6',              # Purple - reference
    'values': '#1abc9c',           # Teal - values
    'adv': '#34495e',              # Dark gray - advantage
    'update_critic': '#e91e63',    # Pink - critic update
    'update_actor': '#c0392b',     # Dark red - actor update
}

# Operation name mapping (short timer name -> readable name)
OPERATION_NAMES = {
    'gen': 'Generation E2E',
    'generate_sequences': 'Token Generation',
    'reward': 'Reward Scoring',
    'old_log_prob': 'Policy Log-Prob',
    'ref': 'Reference Log-Prob',
    'values': 'Value Estimation',
    'adv': 'Advantage Calculation',
    'update_critic': 'Critic Update',
    'update_actor': 'Actor Update',
}

# Which phase each operation belongs to
OPERATION_PHASE = {
    'gen': 'rollout',
    'generate_sequences': 'rollout',
    'reward': 'rl_policy',
    'old_log_prob': 'rl_policy',
    'ref': 'rl_policy',
    'values': 'rl_policy',
    'adv': 'rl_policy',
    'update_critic': 'training',
    'update_actor': 'training',
}

# Phase colors (for backgrounds/annotations)
PHASE_COLORS = {
    'rollout': '#4A90E2',
    'rl_policy': '#F5A623',
    'training': '#D0021B',
}

# Operations to exclude from analysis (profiling overhead and statistics)
EXCLUDED_OPERATIONS = {
    'start_profile',               # Profiling overhead
    'generation_timing/max',       # Statistics, not operations
    'generation_timing/min',
    'generation_timing/topk_ratio',
    'step',                        # Total step time (redundant)
}


class SubPhaseAnalyzer:
    """Analyze sub-phase level GPU profiling data."""
    
    def __init__(self, gpu_csv_path, timing_log_path, output_dir='analysis_output_subphase'):
        """
        Initialize analyzer with GPU metrics and timing data.
        
        Args:
            gpu_csv_path: Path to GPU metrics CSV (from monitor)
            timing_log_path: Path to timing log JSONL file
            output_dir: Directory to save outputs
        """
        self.gpu_csv_path = Path(gpu_csv_path)
        self.timing_log_path = Path(timing_log_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading GPU metrics from: {gpu_csv_path}")
        self.gpu_df = pd.read_csv(gpu_csv_path)
        
        print(f"Loading timing data from: {timing_log_path}")
        self.timing_df = self._load_timing_log(timing_log_path)
        
        self.experiment_name = self.gpu_csv_path.stem
        
        print(f"✓ Loaded {len(self.gpu_df)} GPU samples")
        print(f"✓ Loaded {len(self.timing_df)} timing entries")
        
        # Compute derived metrics
        self._compute_derived_metrics()
    
    def _load_timing_log(self, log_path):
        """Load JSONL timing log into DataFrame."""
        timing_records = []
        with open(log_path, 'r') as f:
            for line in f:
                timing_records.append(json.loads(line))
        return pd.DataFrame(timing_records)
    
    def _compute_derived_metrics(self):
        """Calculate additional metrics and parse timestamps."""
        # GPU metrics
        self.gpu_df['memory_used_gb'] = self.gpu_df['memory_used_mb'] / 1024
        self.gpu_df['elapsed_minutes'] = self.gpu_df['elapsed_seconds'] / 60
        
        # Parse timestamp to Unix time for correlation
        # CSV timestamps are in format: 2025-11-23_21:58:13
        self.gpu_df['timestamp_unix'] = self.gpu_df['timestamp'].apply(
            lambda x: datetime.strptime(x, '%Y-%m-%d_%H:%M:%S').timestamp()
        )
        
        # Timing data - identify which operations occurred (excluding profiling overhead)
        timing_cols = [col for col in self.timing_df.columns 
                      if col not in ['iteration', 'phase', 'timestamp', 'step'] 
                      and col not in EXCLUDED_OPERATIONS]
        self.operation_names = timing_cols
        print(f"✓ Found {len(self.operation_names)} timed operations: {self.operation_names}")
    
    def correlate_timing_with_gpu(self):
        """
        Correlate timing data with GPU metrics using timestamps.
        
        For each operation, we find GPU samples that occurred during the operation's
        execution window (estimated from cumulative timing within each phase).
        """
        correlations = []
        
        for idx, timing_row in self.timing_df.iterrows():
            iteration = timing_row['iteration']
            phase = timing_row['phase']
            phase_end_time = timing_row['timestamp']  # Unix timestamp when phase ended
            
            # Build timeline of operations within this phase
            # Operations are sequential, so we work backwards from phase end time
            operation_times = []
            cumulative_time = 0
            
            for op_name in reversed(self.operation_names):
                if op_name in timing_row and pd.notna(timing_row[op_name]) and timing_row[op_name] > 0:
                    duration = timing_row[op_name]
                    op_end_time = phase_end_time - cumulative_time
                    op_start_time = op_end_time - duration
                    
                    operation_times.append({
                        'operation': op_name,
                        'duration': duration,
                        'start_time': op_start_time,
                        'end_time': op_end_time
                    })
                    
                    cumulative_time += duration
            
            # Reverse back to chronological order
            operation_times = list(reversed(operation_times))
            
            # For each operation, find matching GPU samples
            for op_info in operation_times:
                op_name = op_info['operation']
                start_time = op_info['start_time']
                end_time = op_info['end_time']
                duration = op_info['duration']
                
                # Find GPU samples within this time window (with 1s tolerance)
                mask = (
                    (self.gpu_df['timestamp_unix'] >= start_time - 1) &
                    (self.gpu_df['timestamp_unix'] <= end_time + 1) &
                    (self.gpu_df['phase_name'] == phase) &
                    (self.gpu_df['iteration'] == iteration)
                )
                gpu_samples = self.gpu_df[mask]
                
                # Fallback: if timestamp matching fails, use phase+iteration matching
                if len(gpu_samples) == 0:
                    mask = (self.gpu_df['phase_name'] == phase) & (self.gpu_df['iteration'] == iteration)
                    gpu_samples = self.gpu_df[mask]
                
                if len(gpu_samples) > 0:
                    # Compute average GPU metrics during this operation
                    correlation = {
                        'iteration': iteration,
                        'phase': phase,
                        'operation': op_name,
                        'duration_s': duration,
                        'start_time': start_time,
                        'end_time': end_time,
                        'avg_power_w': gpu_samples['power_draw_w'].mean(),
                        'avg_gpu_util': gpu_samples['gpu_util_percent'].mean(),
                        'avg_temp_c': gpu_samples['temperature_c'].mean(),
                        'avg_memory_gb': gpu_samples['memory_used_gb'].mean(),
                        'samples': len(gpu_samples),
                    }
                    
                    # Calculate energy
                    correlation['energy_j'] = correlation['avg_power_w'] * duration
                    
                    correlations.append(correlation)
                else:
                    print(f"Warning: No GPU samples for {op_name} in iter {iteration}, phase {phase}")
        
        self.correlated_df = pd.DataFrame(correlations)
        print(f"✓ Correlated {len(self.correlated_df)} operation instances")
        
        if len(self.correlated_df) == 0:
            print("\n⚠️  WARNING: No correlations found!")
            print("This usually means timestamp mismatch between CSV and JSONL.")
            print("Debug info:")
            print(f"  CSV timestamp range: {self.gpu_df['timestamp_unix'].min()} - {self.gpu_df['timestamp_unix'].max()}")
            print(f"  JSONL timestamp range: {self.timing_df['timestamp'].min()} - {self.timing_df['timestamp'].max()}")
        
        return self.correlated_df
    
    def compute_subphase_statistics(self):
        """Generate statistics per operation."""
        if not hasattr(self, 'correlated_df'):
            self.correlate_timing_with_gpu()
        
        if len(self.correlated_df) == 0:
            print("⚠️  No correlated data - cannot compute statistics")
            return pd.DataFrame()
        
        stats = []
        
        for op_name in self.correlated_df['operation'].unique():
            op_df = self.correlated_df[self.correlated_df['operation'] == op_name]
            
            if len(op_df) == 0:
                continue
            
            stat = {
                'operation': OPERATION_NAMES.get(op_name, op_name),
                'operation_key': op_name,
                'occurrences': len(op_df),
                'total_duration_s': op_df['duration_s'].sum(),
                'avg_duration_s': op_df['duration_s'].mean(),
                'std_duration_s': op_df['duration_s'].std(),
                'avg_power_w': op_df['avg_power_w'].mean(),
                'std_power_w': op_df['avg_power_w'].std(),
                'avg_gpu_util': op_df['avg_gpu_util'].mean(),
                'std_gpu_util': op_df['avg_gpu_util'].std(),
                'avg_temp_c': op_df['avg_temp_c'].mean(),
                'total_energy_j': op_df['energy_j'].sum(),
                'total_energy_wh': op_df['energy_j'].sum() / 3600,
            }
            
            stats.append(stat)
        
        if len(stats) == 0:
            print("⚠️  No statistics computed")
            return pd.DataFrame()
        
        stats_df = pd.DataFrame(stats)
        stats_df = stats_df.sort_values('total_energy_j', ascending=False)
        return stats_df
    
    def save_subphase_statistics(self, output_path=None):
        """Save sub-phase statistics to CSV."""
        if output_path is None:
            output_path = self.output_dir / f'{self.experiment_name}_subphase_stats.csv'
        
        stats_df = self.compute_subphase_statistics()
        
        if len(stats_df) == 0:
            print("\n⚠️  No statistics to save - correlation failed")
            return stats_df
        
        stats_df.to_csv(output_path, index=False)
        
        print(f"\n✓ Sub-phase statistics saved to: {output_path}")
        print("\n" + "="*70)
        print("SUB-PHASE LEVEL SUMMARY (Ranked by Energy Consumption)")
        print("="*70)
        for _, row in stats_df.iterrows():
            print(f"\n{row['operation']}:")
            print(f"  Occurrences: {row['occurrences']}")
            print(f"  Total Duration: {row['total_duration_s']:.2f}s (avg: {row['avg_duration_s']:.2f}s)")
            print(f"  Avg Power: {row['avg_power_w']:.1f}W (±{row['std_power_w']:.1f}W)")
            print(f"  Avg GPU Util: {row['avg_gpu_util']:.1f}% (±{row['std_gpu_util']:.1f}%)")
            print(f"  Total Energy: {row['total_energy_wh']:.3f}Wh ({row['total_energy_j']:.1f}J)")
        print("="*70)
        
        return stats_df
    
    def plot_operation_energy_breakdown(self):
        """Create separate pie charts for each metric."""
        stats_df = self.compute_subphase_statistics()
        
        if len(stats_df) == 0:
            print("⚠️  Skipping pie charts - no data")
            return
        
        # Create subfolder for pie charts
        pie_folder = self.output_dir / 'pie_charts'
        pie_folder.mkdir(exist_ok=True)
        
        # 5 metrics to visualize
        metrics = [
            ('total_energy_wh', 'Energy Consumption (Wh)', 'energy'),
            ('avg_power_w', 'Average Power (W)', 'power'),
            ('avg_gpu_util', 'Average GPU Utilization (%)', 'gpu_util'),
            ('avg_temp_c', 'Average Temperature (°C)', 'temperature'),
            ('total_duration_s', 'Total Duration (s)', 'duration'),
        ]
        
        for metric_col, title, metric_name in metrics:
            save_path = pie_folder / f'{self.experiment_name}_{metric_name}_pie.png'
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Get data
            operations = stats_df['operation'].values
            values = stats_df[metric_col].values
            colors = [OPERATION_COLORS.get(k, '#95a5a6') for k in stats_df['operation_key'].values]
            
            # Add phase label to operation names
            labels = []
            for op_key, op_name in zip(stats_df['operation_key'].values, operations):
                phase = OPERATION_PHASE.get(op_key, 'unknown')
                labels.append(f"{op_name}\n[{phase}]")
            
            wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                              colors=colors, startangle=90,
                                              textprops={'fontsize': 9},
                                              pctdistance=0.85)
            
            ax.set_title(f'{title} by Operation', fontsize=13, fontweight='bold', pad=20)
            
            # Style percentage labels
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
            
            # Style operation labels
            for text in texts:
                text.set_fontsize(8)
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Pie charts saved to: {pie_folder}/")
    
    def plot_operation_comparison(self, save_path=None):
        """Create box plots comparing 5 metrics across operations."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_operation_comparison.png'
        
        if not hasattr(self, 'correlated_df'):
            self.correlate_timing_with_gpu()
        
        if len(self.correlated_df) == 0:
            print("⚠️  Skipping operation comparison plot - no data")
            return
        
        # Add readable names and phase info
        self.correlated_df['operation_name'] = self.correlated_df['operation'].map(
            lambda x: OPERATION_NAMES.get(x, x)
        )
        self.correlated_df['operation_phase'] = self.correlated_df['operation'].map(
            lambda x: OPERATION_PHASE.get(x, 'unknown')
        )
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Operation-Level Metric Comparison: {self.experiment_name}', 
                     fontsize=14, fontweight='bold')
        
        # 5 metrics to compare
        metrics = [
            ('avg_power_w', 'Power (W)', axes[0, 0]),
            ('avg_gpu_util', 'GPU Utilization (%)', axes[0, 1]),
            ('avg_temp_c', 'Temperature (°C)', axes[0, 2]),
            ('avg_memory_gb', 'Memory Used (GB)', axes[1, 0]),
            ('energy_j', 'Energy (Joules)', axes[1, 1]),
        ]
        
        for metric_col, ylabel, ax in metrics:
            sns.boxplot(data=self.correlated_df, x='operation', y=metric_col, ax=ax,
                       palette=[OPERATION_COLORS.get(op, '#95a5a6') 
                               for op in self.correlated_df['operation'].unique()])
            
            # Set readable x-tick labels with phase
            unique_ops = self.correlated_df['operation'].unique()
            labels = [f"{OPERATION_NAMES.get(op, op)}\n[{OPERATION_PHASE.get(op, '?')}]" 
                     for op in unique_ops]
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            
            ax.set_xlabel('')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(ylabel, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Remove unused subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Operation comparison saved to: {save_path}")
        plt.close()
    
    def plot_operation_timeline(self, save_path=None, num_iterations=10):
        """Create Gantt-like timeline showing operation sequence."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_operation_timeline.png'
        
        if len(self.timing_df) == 0:
            print("⚠️  Skipping timeline plot - no timing data")
            return
        
        # Get timing data from MIDDLE iterations (skip warmup)
        total_iters = len(self.timing_df['iteration'].unique())
        start_iter = max(10, total_iters // 3)  # Start after warmup (1/3 through or iter 10)
        
        # Get iterations in the middle range
        middle_iters = sorted(self.timing_df['iteration'].unique())[start_iter:start_iter + num_iterations]
        timeline_df = self.timing_df[self.timing_df['iteration'].isin(middle_iters)]
        
        if len(timeline_df) == 0:
            print(f"⚠️  No data for middle iterations {start_iter}-{start_iter+num_iterations}")
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        y_pos = 0
        for idx, row in timeline_df.iterrows():
            iteration = row['iteration']
            phase = row['phase']
            
            cumulative_time = 0
            for op_name in self.operation_names:
                if op_name in row and pd.notna(row[op_name]) and row[op_name] > 0:
                    duration = row[op_name]
                    color = OPERATION_COLORS.get(op_name, '#95a5a6')
                    
                    # Use readable operation name
                    readable_name = OPERATION_NAMES.get(op_name, op_name)
                    
                    ax.barh(y_pos, duration, left=cumulative_time, height=0.8,
                           color=color, edgecolor='black', linewidth=0.5,
                           label=readable_name if idx == 0 else "")
                    
                    cumulative_time += duration
            
            ax.text(-0.5, y_pos, f"Iter {iteration}\n{phase}", ha='right', va='center', fontsize=8)
            y_pos += 1
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=2, fontsize=9)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Iteration + Phase', fontsize=12)
        ax.set_title(f'Operation Timeline (Iterations {min(middle_iters)}-{max(middle_iters)})', 
                     fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Operation timeline saved to: {save_path}")
        plt.close()
    
    def plot_per_phase_breakdown(self, save_path=None):
        """Create separate energy breakdown for each phase."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_per_phase_breakdown.png'
        
        if not hasattr(self, 'correlated_df'):
            self.correlate_timing_with_gpu()
        
        if len(self.correlated_df) == 0:
            print("⚠️  Skipping per-phase breakdown - no data")
            return
        
        phases = self.correlated_df['phase'].unique()
        n_phases = len(phases)
        
        fig, axes = plt.subplots(1, n_phases, figsize=(6*n_phases, 6))
        if n_phases == 1:
            axes = [axes]
        
        fig.suptitle(f'Energy Breakdown by Phase: {self.experiment_name}',
                     fontsize=14, fontweight='bold')
        
        for idx, phase in enumerate(phases):
            phase_data = self.correlated_df[self.correlated_df['phase'] == phase]
            
            # Group by operation and sum energy
            op_energy = phase_data.groupby('operation')['energy_j'].sum()
            op_energy_wh = op_energy / 3600
            
            operations = [OPERATION_NAMES.get(op, op) for op in op_energy.index]
            colors = [OPERATION_COLORS.get(op, '#95a5a6') for op in op_energy.index]
            
            wedges, texts, autotexts = axes[idx].pie(op_energy_wh.values, 
                                                     labels=operations,
                                                     autopct='%1.1f%%',
                                                     colors=colors,
                                                     startangle=90,
                                                     textprops={'fontsize': 8},
                                                     pctdistance=0.85)
            
            axes[idx].set_title(f'{phase.upper()} Phase\n({op_energy_wh.sum():.2f} Wh total)',
                               fontsize=11)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            
            for text in texts:
                text.set_fontsize(7)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Per-phase breakdown saved to: {save_path}")
        plt.close()
    
    def plot_operation_correlations(self):
        """Create correlation matrices for each operation (5 metrics like phase analysis)."""
        if not hasattr(self, 'correlated_df'):
            self.correlate_timing_with_gpu()
        
        if len(self.correlated_df) == 0:
            print("⚠️  Skipping correlation matrices - no data")
            return
        
        # Create subfolder
        corr_folder = self.output_dir / 'operation_correlations'
        corr_folder.mkdir(exist_ok=True)
        
        # 5 metrics to correlate (like phase analysis)
        metrics = ['avg_power_w', 'avg_gpu_util', 'avg_temp_c', 'avg_memory_gb', 'avg_mem_bw_gb']
        metric_labels = ['Power (W)', 'GPU Util (%)', 'Temp (°C)', 'Mem Used (GB)', 'Mem BW (GB/s)']
        
        # For each operation, create correlation matrix
        operations = self.correlated_df['operation'].unique()
        
        for op in operations:
            op_data = self.correlated_df[self.correlated_df['operation'] == op]
            
            if len(op_data) < 2:
                continue
            
            # Check which metrics have data
            available_metrics = []
            available_labels = []
            for metric, label in zip(metrics, metric_labels):
                if metric in op_data.columns and op_data[metric].notna().sum() > 1:
                    available_metrics.append(metric)
                    available_labels.append(label)
            
            if len(available_metrics) < 2:
                continue
            
            # Compute correlation
            corr_matrix = op_data[available_metrics].corr()
            
            # Create plot
            readable_name = OPERATION_NAMES.get(op, op)
            phase = OPERATION_PHASE.get(op, 'unknown')
            
            fig, ax = plt.subplots(figsize=(8, 7))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, vmin=-1, vmax=1, square=True, ax=ax,
                       xticklabels=available_labels, yticklabels=available_labels,
                       cbar_kws={'label': 'Correlation Coefficient'})
            
            ax.set_title(f'Metric Correlations: {readable_name}\n[{phase} phase]',
                        fontsize=12, fontweight='bold', pad=15)
            
            plt.tight_layout()
            save_path = corr_folder / f'{self.experiment_name}_corr_{op}.png'
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Operation correlation matrices saved to: {corr_folder}/")
    
    def generate_all_visualizations(self):
        """Generate all sub-phase analysis visualizations."""
        print("\n" + "="*70)
        print("GENERATING SUB-PHASE LEVEL VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Correlate data first
        print("Correlating timing data with GPU metrics...")
        self.correlate_timing_with_gpu()
        
        if len(self.correlated_df) == 0:
            print("\n⚠️  No correlated data - cannot generate visualizations")
            print("Check that GPU CSV and timing log have overlapping iterations/phases")
            return
        
        # Save correlated data
        correlated_csv = self.output_dir / f'{self.experiment_name}_correlated.csv'
        self.correlated_df.to_csv(correlated_csv, index=False)
        print(f"✓ Correlated data saved to: {correlated_csv}")
        
        # Generate statistics and plots
        print("\nGenerating statistics and visualizations...")
        self.save_subphase_statistics()
        
        print("\nGenerating main plots...")
        self.plot_operation_energy_breakdown()  # 5 separate pie charts
        self.plot_operation_comparison()        # Box plots with 5 metrics + phase labels
        self.plot_per_phase_breakdown()         # Energy breakdown per phase
        self.plot_operation_correlations()      # Correlation matrices per operation
        self.plot_operation_timeline()          # Gantt chart
        
        print("\n" + "="*70)
        print("✓ ALL SUB-PHASE VISUALIZATIONS COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {self.output_dir.absolute()}")
        print(f"\nGenerated files and folders:")
        for f in sorted(self.output_dir.glob('*')):
            if f.is_dir():
                num_files = len(list(f.glob('*')))
                print(f"  📁 {f.name}/ ({num_files} files)")
            else:
                print(f"  📄 {f.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sub-phase level GPU profiling data from RLHF training'
    )
    parser.add_argument('--gpu-csv', type=str, required=True,
                       help='Path to GPU metrics CSV file')
    parser.add_argument('--timing-log', type=str, required=True,
                       help='Path to timing log JSONL file')
    parser.add_argument('--output-dir', type=str, default='analysis_output_subphase',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = SubPhaseAnalyzer(args.gpu_csv, args.timing_log, args.output_dir)
    analyzer.generate_all_visualizations()
    
    print("\n✓ Sub-phase analysis complete!\n")


if __name__ == '__main__':
    main()