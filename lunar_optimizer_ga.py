#!/usr/bin/env python3
"""
Genetic Algorithm optimizer for the Lunar Lander game using the actual lunar executable.
Finds optimal fuel rate sequences to achieve the softest landing.

This script uses a genetic algorithm with real lunar executable evaluation to evolve
fuel rate sequences that minimize impact velocity. The algorithm maintains a population
of candidate solutions, selects the best performers, and creates new generations through
crossover and mutation operations.

Example usage:
    # Quick test run
    python lunar_optimizer_ga.py -p 20 -g 30 -s 25
    
    # Balanced optimization
    python lunar_optimizer_ga.py -p 40 -g 50 -m 0.15 -s 30
    
    # Thorough search
    python lunar_optimizer_ga.py -p 60 -g 100 -m 0.1 -c 0.9 -s 35
    
    # Custom output file
    python lunar_optimizer_ga.py -p 30 -g 40 -o my_solution.txt
"""

import random
import copy
import subprocess
import tempfile
import os
import re
import argparse
from typing import List, Tuple
import statistics


class LunarExecutor:
    """Calls the actual lunar executable to evaluate fuel sequences"""

    def __init__(self, lunar_path="./lunar"):
        self.lunar_path = lunar_path
        # Verify lunar executable exists
        if not os.path.exists(lunar_path):
            raise FileNotFoundError(
                f"Lunar executable not found at {lunar_path}")

    def simulate(self, fuel_rates: List[float]) -> Tuple[float, float, bool]:
        """
        Run lunar executable with given fuel rates.
        Returns (impact_velocity_mph, fuel_remaining, crashed)
        """
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            # Write fuel rates
            for rate in fuel_rates:
                f.write(f"{rate}\n")
            # Add extra zeros to ensure we have enough input
            for _ in range(50):  # Add plenty of zeros
                f.write("0\n")
            f.write("NO\n")  # Don't play again
            temp_file = f.name

        try:
            # Run lunar with input redirection
            result = subprocess.run(
                [self.lunar_path, "--echo"],
                stdin=open(temp_file, 'r'),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30
            )

            output = result.stdout

            # Parse the output to extract results
            return self._parse_output(output)

        except subprocess.TimeoutExpired:
            return 9999.0, 0.0, True  # Treat timeout as crash
        except Exception as e:
            print(f"Error running lunar: {e}")
            return 9999.0, 0.0, True
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _parse_output(self, output: str) -> Tuple[float, float, bool]:
        """Parse lunar output to extract impact velocity and fuel remaining"""
        lines = output.strip().split('\n')

        impact_velocity = 9999.0
        fuel_remaining = 0.0
        crashed = True

        for line in lines:
            # Look for "ON THE MOON AT" line
            if "ON THE MOON AT" in line:
                crashed = False

            # Look for "IMPACT VELOCITY OF" line
            impact_match = re.search(
                r"IMPACT VELOCITY OF\s+([\d.]+)\s+M\.P\.H\.", line)
            if impact_match:
                impact_velocity = float(impact_match.group(1))

            # Look for "FUEL LEFT:" line
            fuel_match = re.search(r"FUEL LEFT:\s+([\d.]+)\s+LBS", line)
            if fuel_match:
                fuel_remaining = float(fuel_match.group(1))

            # Check for crash indicators
            if any(crash_phrase in line for crash_phrase in [
                "NO SURVIVORS", "BLEW IT", "CRATER", "CRASH LANDING"
            ]):
                crashed = True

        return impact_velocity, fuel_remaining, crashed


class GeneticOptimizer:
    """Genetic Algorithm to optimize lunar lander fuel sequences using real executable"""

    def __init__(self, population_size=50, mutation_rate=0.1, crossover_rate=0.8):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.executor = LunarExecutor()

        # Start with shorter sequences since real execution is slower
        self.chromosome_length = 25  # Enough for most games

    def create_individual(self) -> List[float]:
        """Create a random fuel rate sequence"""
        individual = []
        for _ in range(self.chromosome_length):
            # Random fuel rate: 0 or 8-200
            if random.random() < 0.4:  # 40% chance of no thrust
                individual.append(0.0)
            else:
                individual.append(random.uniform(8.0, 200.0))
        return individual

    def fitness(self, individual: List[float]) -> float:
        """
        Fitness function using real lunar executable - lower is better
        Prioritizes: 1) Survival, 2) Low impact velocity, 3) Fuel efficiency
        """
        try:
            impact_velocity, fuel_remaining, crashed = self.executor.simulate(
                individual)

            if crashed:
                return 10000 + impact_velocity  # Heavy penalty for crashing

            # Reward soft landings heavily
            fitness_score = impact_velocity

            # Bonus for perfect landing (≤ 1 mph)
            if impact_velocity <= 1.0:
                fitness_score -= 100

            # Small bonus for fuel efficiency
            fitness_score -= fuel_remaining * 0.001

            return max(0, fitness_score)

        except Exception as e:
            print(f"Error evaluating individual: {e}")
            return 10000.0  # Penalty for evaluation errors

    def selection(self, population: List[List[float]], fitnesses: List[float]) -> List[float]:
        """Tournament selection"""
        tournament_size = 3
        tournament_indices = random.sample(
            range(len(population)), tournament_size)
        best_idx = min(tournament_indices, key=lambda i: fitnesses[i])
        return copy.deepcopy(population[best_idx])

    def crossover(
            self,
            parent1: List[float],
            parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)

        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def mutate(self, individual: List[float]) -> List[float]:
        """Gaussian mutation"""
        mutated = copy.deepcopy(individual)

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if mutated[i] == 0:
                    # Maybe turn on thrust
                    if random.random() < 0.5:
                        mutated[i] = random.uniform(8.0, 200.0)
                else:
                    # Gaussian noise or turn off
                    if random.random() < 0.1:
                        mutated[i] = 0.0
                    else:
                        mutated[i] += random.gauss(0, 20)
                        mutated[i] = max(0, min(200, mutated[i]))
                        if 0 < mutated[i] < 8:
                            mutated[i] = 8.0

        return mutated

    def optimize(self, generations=100) -> Tuple[List[float], float]:
        """Run genetic algorithm optimization using real lunar executable"""
        print("Using real lunar executable for fitness evaluation...")
        print("This will be slower but more accurate!")

        # Initialize population
        population = [self.create_individual()
                      for _ in range(self.population_size)]

        best_individual = None
        best_fitness = float('inf')

        for generation in range(generations):
            print(f"Generation {generation + 1}/{generations}...")

            # Evaluate fitness (this is the slow part now)
            fitnesses = []
            for i, individual in enumerate(population):
                fitness = self.fitness(individual)
                fitnesses.append(fitness)
                if i % 10 == 0:
                    print(f"  Evaluated {i + 1}/{len(population)} individuals")

            # Track best solution
            min_fitness_idx = fitnesses.index(min(fitnesses))
            if fitnesses[min_fitness_idx] < best_fitness:
                best_fitness = fitnesses[min_fitness_idx]
                best_individual = copy.deepcopy(population[min_fitness_idx])

            # Print progress
            avg_fitness = statistics.mean(fitnesses)
            print(
                f"Generation {generation}: Best={best_fitness:.2f}, Avg={avg_fitness:.2f}")

            # Test best individual
            impact, fuel, crashed = self.executor.simulate(best_individual)
            status = "CRASHED" if crashed else "LANDED"
            print(
                f"  Best solution: {impact:.2f} mph, {fuel:.0f} lbs fuel, {status}")

            # Create next generation
            new_population = []

            # Keep best individual (elitism)
            new_population.append(copy.deepcopy(best_individual))

            # Generate rest of population
            while len(new_population) < self.population_size:
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

        return best_individual, best_fitness


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm optimizer for the Lunar Lander game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-p", "--population-size",
        type=int,
        default=20,
        help="Size of the genetic algorithm population"
    )

    parser.add_argument(
        "-m", "--mutation-rate",
        type=float,
        default=0.15,
        help="Mutation rate for genetic algorithm (0.0 to 1.0)"
    )

    parser.add_argument(
        "-g", "--generations",
        type=int,
        default=50,
        help="Number of generations to evolve"
    )

    parser.add_argument(
        "-c", "--crossover-rate",
        type=float,
        default=0.8,
        help="Crossover rate for genetic algorithm (0.0 to 1.0)"
    )

    parser.add_argument(
        "-l", "--lunar-path",
        type=str,
        default="./lunar",
        help="Path to the lunar executable"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="optimized_input_ga.txt",
        help="Output filename for the optimized fuel sequence"
    )

    parser.add_argument(
        "-s", "--sequence-length",
        type=int,
        default=25,
        help="Length of the fuel rate sequence (chromosome length)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.population_size < 2:
        parser.error("Population size must be at least 2")

    if not 0.0 <= args.mutation_rate <= 1.0:
        parser.error("Mutation rate must be between 0.0 and 1.0")

    if not 0.0 <= args.crossover_rate <= 1.0:
        parser.error("Crossover rate must be between 0.0 and 1.0")

    if args.generations < 1:
        parser.error("Generations must be at least 1")

    if not os.path.exists(args.lunar_path):
        parser.error(f"Lunar executable not found at: {args.lunar_path}")

    if args.sequence_length < 1:
        parser.error("Sequence length must be at least 1")

    return args


def main():
    """Run the optimization using real lunar executable"""
    args = parse_arguments()

    print("Lunar Lander Genetic Algorithm Optimizer (Real Executable)")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Population size: {args.population_size}")
    print(f"  Mutation rate: {args.mutation_rate}")
    print(f"  Crossover rate: {args.crossover_rate}")
    print(f"  Generations: {args.generations}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Lunar executable: {args.lunar_path}")
    print(f"  Output file: {args.output}")
    print()

    optimizer = GeneticOptimizer(
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate
    )
    optimizer.executor.lunar_path = args.lunar_path
    optimizer.chromosome_length = args.sequence_length

    best_solution, best_fitness = optimizer.optimize(
        generations=args.generations)

    print("\nOptimization Complete!")
    print("=" * 30)

    # Test the best solution one more time
    executor = LunarExecutor()
    impact_velocity, fuel_remaining, crashed = executor.simulate(best_solution)

    print(f"Best fuel sequence: {[round(k, 1) for k in best_solution[:15]]}")
    print(f"Impact velocity: {impact_velocity:.2f} mph")
    print(f"Fuel remaining: {fuel_remaining:.0f} lbs")
    print(f"Result: {'CRASHED' if crashed else 'SUCCESSFUL LANDING'}")

    if impact_velocity <= 1:
        print("🎉 PERFECT LANDING! 🎉")
    elif impact_velocity <= 10:
        print("✅ GOOD LANDING")
    elif impact_velocity <= 22:
        print("⚠️  POOR LANDING")
    else:
        print("💥 CRASH LANDING")

    # Generate input file for manual testing (matching what optimizer tested)
    with open(args.output, 'w') as f:
        # Write the actual fuel rates tested (EXACT precision used in simulation)
        for rate in best_solution:
            f.write(f"{rate}\n")  # Full precision, no rounding
        # Add the same padding zeros used during optimization
        for _ in range(50):
            f.write("0\n")
        f.write("NO\n")

    print(f"\nGenerated '{args.output}' for testing with: "
          f"./lunar < {args.output}")


if __name__ == "__main__":
    main()
