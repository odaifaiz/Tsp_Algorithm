#!/usr/bin/env python3
"""
=============================================================
  TSP Solver — Travelling Salesman Problem
  Algorithms: Brute Force | Nearest Neighbor |
              Simulated Annealing | Genetic Algorithm
=============================================================
"""

import sys
import os
import numpy as np
import random
import time
from tabulate import tabulate
from colorama import Fore, Style, init

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(__file__))

from algorithms.tsp_solvers import TSPSolver
from visualization.plotter import TSPPlotter

init(autoreset=True)

BANNER = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║         Travelling Salesman Problem (TSP) Solver         ║
║  Algorithms: BF | NN | Simulated Annealing | Genetic     ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""

def generate_cities(n: int, seed: int = 42) -> np.ndarray:
    """Generate n random cities in a 100×100 grid."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n, 2))


def run_with_convergence(solver: TSPSolver):
    """Run SA and GA with convergence tracking."""
    convergence = {}

    # --- Simulated Annealing convergence ---
    path = list(range(solver.num_cities))
    random.shuffle(path)
    current_dist = solver.calculate_path_distance(path)
    best_dist = current_dist
    best_path = path.copy()
    temp, cooling = 10000, 0.995
    sa_conv = [best_dist]
    for _ in range(10000):
        if temp < 0.1:
            break
        i, j = random.sample(range(solver.num_cities), 2)
        new_path = path.copy()
        new_path[i], new_path[j] = new_path[j], new_path[i]
        new_dist = solver.calculate_path_distance(new_path)
        if new_dist < current_dist or random.random() < np.exp((current_dist - new_dist) / temp):
            path, current_dist = new_path, new_dist
            if new_dist < best_dist:
                best_dist, best_path = new_dist, new_path.copy()
        temp *= cooling
        sa_conv.append(best_dist)
    convergence['Simulated Annealing'] = sa_conv[::50]   # sample every 50 iters

    # --- Genetic Algorithm convergence ---
    def create_ind():
        ind = list(range(solver.num_cities))
        random.shuffle(ind)
        return ind

    def crossover(p1, p2):
        s, e = sorted(random.sample(range(solver.num_cities), 2))
        child = [-1] * solver.num_cities
        child[s:e] = p1[s:e]
        idx = 0
        for i in range(solver.num_cities):
            if child[i] == -1:
                while p2[idx] in child:
                    idx += 1
                child[i] = p2[idx]
        return child

    pop = [create_ind() for _ in range(100)]
    ga_best = float('inf')
    ga_conv = []
    for _ in range(500):
        fits = [1.0 / solver.calculate_path_distance(ind) for ind in pop]
        best_idx = int(np.argmax(fits))
        d = solver.calculate_path_distance(pop[best_idx])
        if d < ga_best:
            ga_best = d
        ga_conv.append(ga_best)
        new_pop = []
        for _ in range(100):
            t1 = max(random.sample(list(zip(pop, fits)), 3), key=lambda x: x[1])[0]
            t2 = max(random.sample(list(zip(pop, fits)), 3), key=lambda x: x[1])[0]
            child = crossover(t1, t2)
            if random.random() < 0.1:
                a, b = random.sample(range(solver.num_cities), 2)
                child[a], child[b] = child[b], child[a]
            new_pop.append(child)
        pop = new_pop
    convergence['Genetic Algorithm'] = ga_conv[::5]   # sample every 5 gens

    return convergence


def print_results_table(results: dict):
    """Print a formatted comparison table."""
    headers = ['Algorithm', 'Distance', 'Time (ms)', 'Quality vs Best']
    best_dist = min(v['distance'] for v in results.values())
    rows = []
    for algo, data in results.items():
        ratio = (data['distance'] / best_dist - 1) * 100
        quality = f'+{ratio:.1f}%' if ratio > 0.01 else 'BEST'
        rows.append([algo, f"{data['distance']:.4f}", f"{data['time']*1000:.2f}", quality])
    print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))


def main():
    print(BANNER)

    # ── Configuration ──────────────────────────────────────────────
    NUM_CITIES = 15
    SEED       = 42
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
    # ───────────────────────────────────────────────────────────────

    print(f"{Fore.YELLOW}[*] Generating {NUM_CITIES} random cities (seed={SEED})...{Style.RESET_ALL}")
    cities = generate_cities(NUM_CITIES, seed=SEED)
    solver = TSPSolver(cities)
    plotter = TSPPlotter(cities, output_dir=OUTPUT_DIR)

    algorithms = {
        'Brute Force':         solver.solve_brute_force,
        'Nearest Neighbor':    solver.solve_nearest_neighbor,
        'Simulated Annealing': solver.solve_simulated_annealing,
        'Genetic Algorithm':   solver.solve_genetic_algorithm,
    }

    results = {}
    print()
    for name, func in algorithms.items():
        print(f"{Fore.CYAN}[*] Running {name}...{Style.RESET_ALL}", end=' ', flush=True)
        path, distance, exec_time = func()
        results[name] = {'path': path, 'distance': distance, 'time': exec_time}
        print(f"{Fore.GREEN}Done  |  Distance: {distance:.2f}  |  Time: {exec_time*1000:.2f} ms{Style.RESET_ALL}")

    print(f"\n{Fore.YELLOW}{'='*60}")
    print("  RESULTS COMPARISON")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    print_results_table(results)

    # ── Plotting ───────────────────────────────────────────────────
    print(f"\n{Fore.YELLOW}[*] Generating visualizations...{Style.RESET_ALL}")

    for name, data in results.items():
        f = plotter.plot_single_solution(data['path'], name, data['distance'], data['time'])
        print(f"  {Fore.GREEN}Saved:{Style.RESET_ALL} {f}")

    f = plotter.plot_comparison(results)
    print(f"  {Fore.GREEN}Saved:{Style.RESET_ALL} {f}")

    f = plotter.plot_performance_chart(results)
    print(f"  {Fore.GREEN}Saved:{Style.RESET_ALL} {f}")

    print(f"\n{Fore.YELLOW}[*] Computing convergence curves (SA & GA)...{Style.RESET_ALL}")
    convergence = run_with_convergence(solver)
    f = plotter.plot_convergence(convergence)
    print(f"  {Fore.GREEN}Saved:{Style.RESET_ALL} {f}")

    print(f"\n{Fore.CYAN}All results saved to: {OUTPUT_DIR}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Done!{Style.RESET_ALL}\n")

    return results


if __name__ == '__main__':
    main()
