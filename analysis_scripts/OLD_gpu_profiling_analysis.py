#!/usr/bin/env python3
"""
GPU Profiling Analysis for RLHF Training
Comprehensive analysis and visualization of GPU metrics during Verl PPO training.

Usage:
    python gpu_profiling_analysis.py <csv_file> [--output-dir <dir>]
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


class GPUProfiler:
    """Analyze and visualize GPU profiling data from RLHF training."""
    
    def __init__(self, csv_path, output_dir='analysis_output'):
        """
        Initialize profiler with CSV data.
        
        Args:
            csv_path: Path to GPU metrics CSV file
            output_dir: Directory to save outputs
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.experiment_name = self.csv_path.stem.replace('_gpu_metrics', '')
        
        # Compute derived metrics
        self._compute_derived_metrics()
        
    def _compute_derived_metrics(self):
        """Calculate additional metrics from raw data."""
        # Memory in GB
        self.df['memory_used_gb'] = self.df['memory_used_mb'] / 1024
        self.df['memory_total_gb'] = self.df['memory_total_mb'] / 1024
        self.df['memory_free_gb'] = self.df['memory_total_gb'] - self.df['memory_used_gb']
        
        # Time in minutes
        self.df['elapsed_minutes'] = self.df['elapsed_seconds'] / 60
        
        # Power efficiency metrics
        self.df['power_efficiency'] = self.df['gpu_util_percent'] / (self.df['power_draw_w'] + 1e-6)
        self.df['thermal_efficiency'] = self.df['gpu_util_percent'] / (self.df['temperature_c'] + 1e-6)
        
        # Utilization ratios
        self.df['power_to_limit_ratio'] = self.df['power_draw_w'] / self.df['power_limit_w']
        self.df['memory_usage_ratio'] = self.df['memory_used_gb'] / self.df['memory_total_gb']
        
    def compute_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        stats = {
            'experiment': self.experiment_name,
            'gpu_name': self.df['gpu_name'].iloc[0],
            'duration_seconds': self.df['elapsed_seconds'].max(),
            'duration_minutes': self.df['elapsed_minutes'].max(),
            'samples_collected': len(self.df),
            'sampling_rate_hz': len(self.df) / (self.df['elapsed_seconds'].max() + 1),
            
            # GPU Utilization
            'gpu_util_mean': self.df['gpu_util_percent'].mean(),
            'gpu_util_std': self.df['gpu_util_percent'].std(),
            'gpu_util_min': self.df['gpu_util_percent'].min(),
            'gpu_util_max': self.df['gpu_util_percent'].max(),
            'gpu_util_median': self.df['gpu_util_percent'].median(),
            'gpu_util_p95': self.df['gpu_util_percent'].quantile(0.95),
            
            # Memory
            'memory_used_mean_gb': self.df['memory_used_gb'].mean(),
            'memory_used_peak_gb': self.df['memory_used_gb'].max(),
            'memory_total_gb': self.df['memory_total_gb'].iloc[0],
            'memory_util_mean': self.df['memory_util_percent'].mean(),
            'memory_util_max': self.df['memory_util_percent'].max(),
            
            # Power
            'power_draw_mean_w': self.df['power_draw_w'].mean(),
            'power_draw_std_w': self.df['power_draw_w'].std(),
            'power_draw_min_w': self.df['power_draw_w'].min(),
            'power_draw_max_w': self.df['power_draw_w'].max(),
            'power_limit_w': self.df['power_limit_w'].iloc[0],
            'power_to_limit_mean': self.df['power_to_limit_ratio'].mean(),
            
            # Thermal
            'temperature_mean_c': self.df['temperature_c'].mean(),
            'temperature_std_c': self.df['temperature_c'].std(),
            'temperature_min_c': self.df['temperature_c'].min(),
            'temperature_max_c': self.df['temperature_c'].max(),
            
            # Clock speeds
            'sm_clock_mean_mhz': self.df['sm_clock_mhz'].mean(),
            'mem_clock_mean_mhz': self.df['mem_clock_mhz'].mean(),
            
            # Efficiency metrics
            'power_efficiency_mean': self.df['power_efficiency'].mean(),
            'thermal_efficiency_mean': self.df['thermal_efficiency'].mean(),
            
            # Energy consumption
            'total_energy_wh': (self.df['power_draw_w'].mean() * 
                               self.df['elapsed_seconds'].max() / 3600),
        }
        
        # Detect idle vs active periods
        active_threshold = 10  # GPU util > 10%
        self.df['is_active'] = self.df['gpu_util_percent'] > active_threshold
        
        active_time = self.df[self.df['is_active']]['elapsed_seconds'].count()
        total_time = len(self.df)
        
        stats['active_samples'] = active_time
        stats['active_ratio'] = active_time / total_time if total_time > 0 else 0
        stats['idle_samples'] = total_time - active_time
        stats['idle_ratio'] = 1 - stats['active_ratio']
        
        if active_time > 0:
            active_df = self.df[self.df['is_active']]
            stats['gpu_util_mean_active'] = active_df['gpu_util_percent'].mean()
            stats['power_draw_mean_active_w'] = active_df['power_draw_w'].mean()
        
        return stats
    
    def save_summary_statistics(self, output_path=None):
        """Save summary statistics to CSV."""
        if output_path is None:
            output_path = self.output_dir / f'{self.experiment_name}_summary_stats.csv'
        
        stats = self.compute_summary_statistics()
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(output_path, index=False)
        
        print(f"✓ Summary statistics saved to: {output_path}")
        return stats_df
    
    def plot_overview(self, save_path=None):
        """Create comprehensive overview plot with key metrics."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_overview.png'
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(f'GPU Profile Overview: {self.experiment_name}', 
                     fontsize=14, fontweight='bold')
        
        time = self.df['elapsed_minutes']
        
        # 1. GPU Utilization
        ax = axes[0, 0]
        ax.plot(time, self.df['gpu_util_percent'], linewidth=1, alpha=0.8, color='#2ecc71')
        ax.fill_between(time, 0, self.df['gpu_util_percent'], alpha=0.3, color='#2ecc71')
        ax.axhline(y=self.df['gpu_util_percent'].mean(), color='red', 
                   linestyle='--', alpha=0.7, label=f"Mean: {self.df['gpu_util_percent'].mean():.1f}%")
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Compute Utilization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # 2. Memory Usage
        ax = axes[0, 1]
        ax.plot(time, self.df['memory_used_gb'], linewidth=1, alpha=0.8, color='#3498db')
        ax.axhline(y=self.df['memory_total_gb'].iloc[0], color='red', 
                   linestyle='--', alpha=0.5, label=f"Total: {self.df['memory_total_gb'].iloc[0]:.1f} GB")
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Memory Used (GB)')
        ax.set_title('GPU Memory Usage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Power Draw
        ax = axes[0, 2]
        ax.plot(time, self.df['power_draw_w'], linewidth=1, alpha=0.8, color='#e74c3c')
        ax.axhline(y=self.df['power_limit_w'].iloc[0], color='orange', 
                   linestyle='--', alpha=0.5, label=f"Limit: {self.df['power_limit_w'].iloc[0]:.0f} W")
        ax.axhline(y=self.df['power_draw_w'].mean(), color='darkred', 
                   linestyle='--', alpha=0.7, label=f"Mean: {self.df['power_draw_w'].mean():.1f} W")
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Power Draw (W)')
        ax.set_title('GPU Power Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Temperature
        ax = axes[1, 0]
        ax.plot(time, self.df['temperature_c'], linewidth=1, alpha=0.8, color='#f39c12')
        ax.axhline(y=self.df['temperature_c'].mean(), color='red', 
                   linestyle='--', alpha=0.7, label=f"Mean: {self.df['temperature_c'].mean():.1f}°C")
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('GPU Temperature')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. SM Clock Speed
        ax = axes[1, 1]
        ax.plot(time, self.df['sm_clock_mhz'], linewidth=1, alpha=0.8, color='#9b59b6')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('SM Clock (MHz)')
        ax.set_title('Streaming Multiprocessor Clock Speed')
        ax.grid(True, alpha=0.3)
        
        # 6. Memory Bandwidth Utilization
        ax = axes[1, 2]
        ax.plot(time, self.df['memory_util_percent'], linewidth=1, alpha=0.8, color='#1abc9c')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Memory Bandwidth Util (%)')
        ax.set_title('Memory Bandwidth Utilization')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Overview plot saved to: {save_path}")
        plt.close()
    
    def plot_efficiency_analysis(self, save_path=None):
        """Analyze power and thermal efficiency."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_efficiency.png'
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Efficiency Analysis: {self.experiment_name}', 
                     fontsize=14, fontweight='bold')
        
        # 1. Power vs GPU Utilization
        ax = axes[0, 0]
        scatter = ax.scatter(self.df['gpu_util_percent'], self.df['power_draw_w'], 
                           c=self.df['elapsed_minutes'], cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('GPU Utilization (%)')
        ax.set_ylabel('Power Draw (W)')
        ax.set_title('Power Consumption vs GPU Utilization')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Time (min)')
        
        # 2. Power Efficiency over time
        ax = axes[0, 1]
        ax.plot(self.df['elapsed_minutes'], self.df['power_efficiency'], 
                linewidth=1, alpha=0.8, color='#16a085')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Efficiency (% util / W)')
        ax.set_title('Power Efficiency (GPU Util / Power)')
        ax.grid(True, alpha=0.3)
        
        # 3. Temperature vs Power
        ax = axes[1, 0]
        scatter = ax.scatter(self.df['power_draw_w'], self.df['temperature_c'], 
                           c=self.df['elapsed_minutes'], cmap='coolwarm', alpha=0.6, s=20)
        ax.set_xlabel('Power Draw (W)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Thermal Response to Power Draw')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Time (min)')
        
        # 4. Utilization Distribution
        ax = axes[1, 1]
        ax.hist(self.df['gpu_util_percent'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        ax.axvline(x=self.df['gpu_util_percent'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f"Mean: {self.df['gpu_util_percent'].mean():.1f}%")
        ax.axvline(x=self.df['gpu_util_percent'].median(), color='orange', 
                   linestyle='--', linewidth=2, label=f"Median: {self.df['gpu_util_percent'].median():.1f}%")
        ax.set_xlabel('GPU Utilization (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('GPU Utilization Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Efficiency analysis saved to: {save_path}")
        plt.close()
    
    def plot_utilization_heatmap(self, save_path=None, window_size=30):
        """Create a heatmap showing utilization patterns over time."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_heatmap.png'
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'Utilization Patterns: {self.experiment_name}', 
                     fontsize=14, fontweight='bold')
        
        # Rolling statistics
        self.df['gpu_util_rolling'] = self.df['gpu_util_percent'].rolling(window=window_size, center=True).mean()
        self.df['power_rolling'] = self.df['power_draw_w'].rolling(window=window_size, center=True).mean()
        self.df['temp_rolling'] = self.df['temperature_c'].rolling(window=window_size, center=True).mean()
        
        time = self.df['elapsed_minutes']
        
        # GPU Utilization with rolling average
        ax = axes[0]
        ax.plot(time, self.df['gpu_util_percent'], alpha=0.3, linewidth=0.5, color='gray', label='Raw')
        ax.plot(time, self.df['gpu_util_rolling'], linewidth=2, color='#2ecc71', label=f'{window_size}s Rolling Avg')
        ax.fill_between(time, 0, self.df['gpu_util_rolling'], alpha=0.3, color='#2ecc71')
        ax.set_ylabel('GPU Util (%)')
        ax.set_title('GPU Utilization with Smoothing')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Power with rolling average
        ax = axes[1]
        ax.plot(time, self.df['power_draw_w'], alpha=0.3, linewidth=0.5, color='gray', label='Raw')
        ax.plot(time, self.df['power_rolling'], linewidth=2, color='#e74c3c', label=f'{window_size}s Rolling Avg')
        ax.set_ylabel('Power (W)')
        ax.set_title('Power Draw with Smoothing')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Temperature with rolling average
        ax = axes[2]
        ax.plot(time, self.df['temperature_c'], alpha=0.3, linewidth=0.5, color='gray', label='Raw')
        ax.plot(time, self.df['temp_rolling'], linewidth=2, color='#f39c12', label=f'{window_size}s Rolling Avg')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature with Smoothing')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        print(f"✓ Utilization heatmap saved to: {save_path}")
        plt.close()
    
    def export_processed_data(self, save_path=None):
        """Export processed dataframe with derived metrics."""
        if save_path is None:
            save_path = self.output_dir / f'{self.experiment_name}_processed_data.csv'
        
        self.df.to_csv(save_path, index=False)
        print(f"✓ Processed data saved to: {save_path}")
    
    def generate_report(self):
        """Generate complete analysis report with all plots and statistics."""
        print("=" * 60)
        print(f"GPU Profiling Report: {self.experiment_name}")
        print("=" * 60)
        print()
        
        # Summary statistics
        stats = self.compute_summary_statistics()
        
        print(f"GPU: {stats['gpu_name']}")
        print(f"Duration: {stats['duration_minutes']:.2f} minutes ({stats['duration_seconds']:.0f} seconds)")
        print(f"Samples: {stats['samples_collected']} @ {stats['sampling_rate_hz']:.2f} Hz")
        print()
        
        print("GPU Utilization:")
        print(f"  Mean: {stats['gpu_util_mean']:.1f}% (±{stats['gpu_util_std']:.1f}%)")
        print(f"  Range: [{stats['gpu_util_min']:.1f}%, {stats['gpu_util_max']:.1f}%]")
        print(f"  Median: {stats['gpu_util_median']:.1f}% | P95: {stats['gpu_util_p95']:.1f}%")
        print(f"  Active time: {stats['active_ratio']*100:.1f}% ({stats['active_samples']} samples > 10% util)")
        if 'gpu_util_mean_active' in stats:
            print(f"  Mean (active only): {stats['gpu_util_mean_active']:.1f}%")
        print()
        
        print("Memory:")
        print(f"  Mean usage: {stats['memory_used_mean_gb']:.2f} GB")
        print(f"  Peak usage: {stats['memory_used_peak_gb']:.2f} GB / {stats['memory_total_gb']:.2f} GB")
        print(f"  Bandwidth util: {stats['memory_util_mean']:.1f}% (mean)")
        print()
        
        print("Power & Thermal:")
        print(f"  Power: {stats['power_draw_mean_w']:.1f}W (±{stats['power_draw_std_w']:.1f}W)")
        print(f"  Power range: [{stats['power_draw_min_w']:.1f}W, {stats['power_draw_max_w']:.1f}W]")
        print(f"  Power limit: {stats['power_limit_w']:.0f}W ({stats['power_to_limit_mean']*100:.1f}% utilized)")
        print(f"  Temperature: {stats['temperature_mean_c']:.1f}°C (±{stats['temperature_std_c']:.1f}°C)")
        print(f"  Total energy: {stats['total_energy_wh']:.2f} Wh")
        print()
        
        print("Clock Speeds:")
        print(f"  SM Clock: {stats['sm_clock_mean_mhz']:.0f} MHz")
        print(f"  Memory Clock: {stats['mem_clock_mean_mhz']:.0f} MHz")
        print()
        
        print("Efficiency:")
        print(f"  Power efficiency: {stats['power_efficiency_mean']:.4f} %util/W")
        print(f"  Thermal efficiency: {stats['thermal_efficiency_mean']:.4f} %util/°C")
        print()
        
        # Generate all plots
        print("Generating visualizations...")
        self.save_summary_statistics()
        self.plot_overview()
        self.plot_efficiency_analysis()
        self.plot_utilization_heatmap()
        self.export_processed_data()
        
        print()
        print("=" * 60)
        print("✓ Analysis complete!")
        print(f"All outputs saved to: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Analyze GPU profiling data from Verl training')
    parser.add_argument('csv_file', type=str, help='Path to GPU metrics CSV file')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    # Run analysis
    profiler = GPUProfiler(args.csv_file, args.output_dir)
    profiler.generate_report()


if __name__ == '__main__':
    main()