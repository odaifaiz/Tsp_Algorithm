import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import List, Dict, Any
import os

class TSPPlotter:
    """Class for visualizing TSP solutions."""
    
    COLORS = {
        'Brute Force':          '#e74c3c',
        'Nearest Neighbor':     '#3498db',
        'Simulated Annealing':  '#2ecc71',
        'Genetic Algorithm':    '#f39c12',
    }
    
    BACKGROUND = '#1a1a2e'
    CITY_COLOR  = '#ffffff'
    GRID_COLOR  = '#2a2a4a'

    def __init__(self, cities: np.ndarray, output_dir: str = 'results'):
        self.cities = cities
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _setup_ax(self, ax, title: str):
        """Apply consistent dark-theme styling to an axes object."""
        ax.set_facecolor(self.BACKGROUND)
        ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=10)
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor(self.GRID_COLOR)
        ax.grid(True, color=self.GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)

    def plot_single_solution(self, path: List[int], algorithm_name: str,
                             distance: float, exec_time: float,
                             save: bool = True) -> str:
        """Plot a single TSP solution."""
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(self.BACKGROUND)
        
        color = self.COLORS.get(algorithm_name, '#9b59b6')
        
        # Draw edges
        full_path = path + [path[0]]
        xs = self.cities[full_path, 0]
        ys = self.cities[full_path, 1]
        ax.plot(xs, ys, '-', color=color, linewidth=1.8, alpha=0.85, zorder=1)
        
        # Draw arrows to show direction
        for i in range(len(full_path) - 1):
            x1, y1 = self.cities[full_path[i]]
            x2, y2 = self.cities[full_path[i+1]]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                        zorder=2)
        
        # Draw cities
        ax.scatter(self.cities[:, 0], self.cities[:, 1],
                   s=80, c=self.CITY_COLOR, zorder=3, edgecolors=color, linewidths=1.5)
        
        # Label cities
        for i, (x, y) in enumerate(self.cities):
            ax.text(x + 0.5, y + 0.5, str(i), color='white', fontsize=8,
                    fontweight='bold', zorder=4)
        
        # Mark start city
        ax.scatter(*self.cities[path[0]], s=180, c='#f1c40f', zorder=5,
                   edgecolors='white', linewidths=2, marker='*', label='Start City')
        
        self._setup_ax(ax, f'{algorithm_name}')
        ax.set_xlabel('X', color='#aaaaaa')
        ax.set_ylabel('Y', color='#aaaaaa')
        
        info_text = f'Total Distance: {distance:.2f}\nTime: {exec_time*1000:.2f} ms\nCities: {len(self.cities)}'
        ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
                color='white', fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#2a2a4a', alpha=0.8))
        
        ax.legend(loc='lower right', facecolor='#2a2a4a', edgecolor='#555555',
                  labelcolor='white', fontsize=9)
        
        plt.tight_layout()
        
        filename = ''
        if save:
            filename = os.path.join(self.output_dir, f'{algorithm_name.lower().replace(" ", "_")}_solution.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.BACKGROUND)
        plt.close()
        return filename

    def plot_comparison(self, results: Dict[str, Dict[str, Any]], save: bool = True) -> str:
        """Plot all algorithm solutions side by side for comparison."""
        n = len(results)
        cols = 2
        rows = (n + 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 6))
        fig.patch.set_facecolor(self.BACKGROUND)
        
        if rows == 1:
            axes = [axes] if n == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]
        
        for idx, (algo_name, data) in enumerate(results.items()):
            ax = axes[idx]
            color = self.COLORS.get(algo_name, '#9b59b6')
            path = data['path']
            
            full_path = path + [path[0]]
            xs = self.cities[full_path, 0]
            ys = self.cities[full_path, 1]
            ax.plot(xs, ys, '-', color=color, linewidth=1.5, alpha=0.8)
            ax.scatter(self.cities[:, 0], self.cities[:, 1],
                       s=60, c=self.CITY_COLOR, zorder=3, edgecolors=color, linewidths=1.2)
            ax.scatter(*self.cities[path[0]], s=140, c='#f1c40f', zorder=5,
                       edgecolors='white', linewidths=1.5, marker='*')
            
            for i, (x, y) in enumerate(self.cities):
                ax.text(x + 0.4, y + 0.4, str(i), color='white', fontsize=7, zorder=4)
            
            self._setup_ax(ax, algo_name)
            
            info = f'Distance: {data["distance"]:.2f}  |  Time: {data["time"]*1000:.2f} ms'
            ax.set_xlabel(info, color='#cccccc', fontsize=9)
        
        # Hide unused axes
        for idx in range(len(results), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('TSP Algorithms Comparison', color='white', fontsize=16,
                     fontweight='bold', y=1.01)
        plt.tight_layout()
        
        filename = ''
        if save:
            filename = os.path.join(self.output_dir, 'comparison_all_algorithms.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.BACKGROUND)
        plt.close()
        return filename

    def plot_performance_chart(self, results: Dict[str, Dict[str, Any]], save: bool = True) -> str:
        """Plot a performance bar chart comparing distances and times."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor(self.BACKGROUND)
        
        algo_names = list(results.keys())
        distances   = [results[a]['distance'] for a in algo_names]
        times_ms    = [results[a]['time'] * 1000 for a in algo_names]
        colors      = [self.COLORS.get(a, '#9b59b6') for a in algo_names]
        
        # Distance chart
        bars1 = ax1.bar(algo_names, distances, color=colors, edgecolor='white',
                        linewidth=0.8, alpha=0.9)
        self._setup_ax(ax1, 'Total Distance Comparison')
        ax1.set_ylabel('Distance (units)', color='#aaaaaa')
        ax1.tick_params(axis='x', rotation=15, colors='#cccccc')
        ax1.tick_params(axis='y', colors='#cccccc')
        for bar, val in zip(bars1, distances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(distances)*0.01,
                     f'{val:.1f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
        
        # Time chart
        bars2 = ax2.bar(algo_names, times_ms, color=colors, edgecolor='white',
                        linewidth=0.8, alpha=0.9)
        self._setup_ax(ax2, 'Execution Time Comparison')
        ax2.set_ylabel('Time (ms)', color='#aaaaaa')
        ax2.tick_params(axis='x', rotation=15, colors='#cccccc')
        ax2.tick_params(axis='y', colors='#cccccc')
        for bar, val in zip(bars2, times_ms):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times_ms)*0.01,
                     f'{val:.2f}', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        filename = ''
        if save:
            filename = os.path.join(self.output_dir, 'performance_chart.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.BACKGROUND)
        plt.close()
        return filename

    def plot_convergence(self, convergence_data: Dict[str, List[float]], save: bool = True) -> str:
        """Plot convergence curves for iterative algorithms."""
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor(self.BACKGROUND)
        
        for algo_name, distances in convergence_data.items():
            color = self.COLORS.get(algo_name, '#9b59b6')
            ax.plot(distances, color=color, linewidth=2, label=algo_name, alpha=0.9)
        
        self._setup_ax(ax, 'Convergence Curve (Distance over Iterations)')
        ax.set_xlabel('Iteration', color='#aaaaaa')
        ax.set_ylabel('Best Distance Found', color='#aaaaaa')
        ax.tick_params(colors='#cccccc')
        ax.legend(facecolor='#2a2a4a', edgecolor='#555555', labelcolor='white', fontsize=10)
        
        plt.tight_layout()
        
        filename = ''
        if save:
            filename = os.path.join(self.output_dir, 'convergence_curves.png')
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=self.BACKGROUND)
        plt.close()
        return filename
