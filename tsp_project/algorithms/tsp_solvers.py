import numpy as np
import itertools
import random
import time
from typing import List, Tuple, Dict, Any

class TSPSolver:
    def __init__(self, cities: np.ndarray):
        """
        Initialize the TSP solver with a set of cities.
        :param cities: numpy array of shape (n, 2) representing (x, y) coordinates of cities.
        """
        self.cities = cities
        self.num_cities = len(cities)
        self.distance_matrix = self._calculate_distance_matrix()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate the Euclidean distance between all pairs of cities."""
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def calculate_path_distance(self, path: List[int]) -> float:
        """Calculate the total distance of a given path."""
        distance = 0.0
        for i in range(len(path) - 1):
            distance += self.distance_matrix[path[i], path[i+1]]
        # Add distance from last city back to the first
        distance += self.distance_matrix[path[-1], path[0]]
        return distance

    def solve_brute_force(self) -> Tuple[List[int], float, float]:
        """
        Solve TSP using Brute Force (Exact but very slow, O(n!)).
        Returns: (best_path, best_distance, execution_time)
        """
        start_time = time.time()
        
        if self.num_cities > 10:
            print(f"Warning: Brute force is limited to ≤10 cities. Running Nearest Neighbor instead.")
            return self.solve_nearest_neighbor()

        cities_indices = list(range(self.num_cities))
        best_path = None
        best_distance = float('inf')

        # Fix the starting city to 0 to reduce permutations by a factor of n
        for perm in itertools.permutations(cities_indices[1:]):
            current_path = [0] + list(perm)
            current_distance = self.calculate_path_distance(current_path)
            
            if current_distance < best_distance:
                best_distance = current_distance
                best_path = current_path

        execution_time = time.time() - start_time
        return best_path, best_distance, execution_time

    def solve_nearest_neighbor(self) -> Tuple[List[int], float, float]:
        """
        Solve TSP using Nearest Neighbor Heuristic (Fast, O(n^2), but not optimal).
        Returns: (best_path, best_distance, execution_time)
        """
        start_time = time.time()
        
        unvisited = set(range(1, self.num_cities))
        current_city = 0
        path = [current_city]
        
        while unvisited:
            nearest_city = min(unvisited, key=lambda city: self.distance_matrix[current_city, city])
            path.append(nearest_city)
            unvisited.remove(nearest_city)
            current_city = nearest_city
            
        distance = self.calculate_path_distance(path)
        execution_time = time.time() - start_time
        
        return path, distance, execution_time

    def solve_simulated_annealing(self, initial_temp: float = 10000, cooling_rate: float = 0.995, iterations: int = 10000) -> Tuple[List[int], float, float]:
        """
        Solve TSP using Simulated Annealing (Metaheuristic).
        Returns: (best_path, best_distance, execution_time)
        """
        start_time = time.time()
        
        # Start with a random path
        current_path = list(range(self.num_cities))
        random.shuffle(current_path)
        current_distance = self.calculate_path_distance(current_path)
        
        best_path = current_path.copy()
        best_distance = current_distance
        
        temp = initial_temp
        
        for _ in range(iterations):
            if temp < 0.1:
                break
                
            # Generate a neighbor by swapping two random cities
            i, j = random.sample(range(self.num_cities), 2)
            new_path = current_path.copy()
            new_path[i], new_path[j] = new_path[j], new_path[i]
            
            new_distance = self.calculate_path_distance(new_path)
            
            # Decide whether to accept the new path
            if new_distance < current_distance:
                current_path = new_path
                current_distance = new_distance
                if new_distance < best_distance:
                    best_path = new_path.copy()
                    best_distance = new_distance
            else:
                # Accept worse path with some probability
                acceptance_probability = np.exp((current_distance - new_distance) / temp)
                if random.random() < acceptance_probability:
                    current_path = new_path
                    current_distance = new_distance
                    
            # Cool down
            temp *= cooling_rate
            
        execution_time = time.time() - start_time
        return best_path, best_distance, execution_time

    def solve_genetic_algorithm(self, population_size: int = 100, generations: int = 500, mutation_rate: float = 0.1) -> Tuple[List[int], float, float]:
        """
        Solve TSP using Genetic Algorithm.
        Returns: (best_path, best_distance, execution_time)
        """
        start_time = time.time()
        
        def create_individual():
            ind = list(range(self.num_cities))
            random.shuffle(ind)
            return ind
            
        def calculate_fitness(individual):
            return 1.0 / self.calculate_path_distance(individual)
            
        def crossover(parent1, parent2):
            # Order Crossover (OX1)
            start, end = sorted(random.sample(range(self.num_cities), 2))
            child = [-1] * self.num_cities
            child[start:end] = parent1[start:end]
            
            p2_idx = 0
            for i in range(self.num_cities):
                if child[i] == -1:
                    while parent2[p2_idx] in child:
                        p2_idx += 1
                    child[i] = parent2[p2_idx]
            return child
            
        def mutate(individual):
            if random.random() < mutation_rate:
                i, j = random.sample(range(self.num_cities), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual

        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        
        best_path = None
        best_distance = float('inf')
        
        for _ in range(generations):
            # Evaluate fitness
            fitness_scores = [calculate_fitness(ind) for ind in population]
            
            # Update best
            max_fitness_idx = np.argmax(fitness_scores)
            current_best_distance = self.calculate_path_distance(population[max_fitness_idx])
            if current_best_distance < best_distance:
                best_distance = current_best_distance
                best_path = population[max_fitness_idx].copy()
                
            # Selection (Tournament)
            new_population = []
            for _ in range(population_size):
                tournament = random.sample(list(zip(population, fitness_scores)), 3)
                parent1 = max(tournament, key=lambda x: x[1])[0]
                tournament = random.sample(list(zip(population, fitness_scores)), 3)
                parent2 = max(tournament, key=lambda x: x[1])[0]
                
                # Crossover & Mutation
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
                
            population = new_population
            
        execution_time = time.time() - start_time
        return best_path, best_distance, execution_time
