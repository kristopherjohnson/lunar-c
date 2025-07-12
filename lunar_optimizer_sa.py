#!/usr/bin/env python3
"""
Simulated Annealing optimizer for the Lunar Lander game using the actual lunar executable.
Finds optimal fuel rate sequences to achieve the softest landing.

This script uses simulated annealing with real lunar executable evaluation to optimize
fuel rate sequences that minimize impact velocity. The algorithm starts with high
"temperature" allowing exploration of the solution space, then gradually cools down
to focus on local optimization of promising solutions.

Example usage:
    # Quick test run
    python lunar_optimizer_sa.py -t 1000 -c 0.95 -i 1000 -s 25
    
    # Balanced optimization  
    python lunar_optimizer_sa.py -t 5000 -c 0.996 -m 0.01 -i 6000 -s 30
    
    # Thorough search (high quality)
    python lunar_optimizer_sa.py -t 8000 -c 0.999 -m 0.001 -i 10000 -s 35
    
    # Multiple restart approach
    python lunar_optimizer_sa.py -t 3000 -c 0.992 -m 0.1 -i 3000 -s 40
    
    # Custom output file
    python lunar_optimizer_sa.py -t 2000 -i 2000 -o my_solution.txt
"""

import random
import copy
import subprocess
import tempfile
import os
import re
import argparse
import math
from typing import List, Tuple


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
            result_tuple = self._parse_output(output)
            return result_tuple

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


class SimulatedAnnealingOptimizer:
    """Simulated Annealing to optimize lunar lander fuel sequences using real executable"""

    def __init__(self, initial_temp=1000.0, cooling_rate=0.95, min_temp=0.1):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.executor = LunarExecutor()

        # Start with shorter sequences since real execution is slower
        self.chromosome_length = 25  # Enough for most games

    def create_random_solution(self) -> List[float]:
        """Create a random fuel rate sequence"""
        solution = []
        for _ in range(self.chromosome_length):
            # Random fuel rate: 0 or 8-200
            if random.random() < 0.4:  # 40% chance of no thrust
                solution.append(0.0)
            else:
                solution.append(random.uniform(8.0, 200.0))
        return solution

    def cost_function(self, solution: List[float]) -> float:
        """
        Cost function using real lunar executable - lower is better
        Prioritizes: 1) Survival, 2) Low impact velocity, 3) Fuel efficiency
        """
        try:
            impact_velocity, fuel_remaining, crashed = self.executor.simulate(
                solution)

            if crashed:
                return 10000 + impact_velocity  # Heavy penalty for crashing

            # Reward soft landings heavily
            cost = impact_velocity

            # Bonus for perfect landing (≤ 1 mph)
            if impact_velocity <= 1.0:
                cost -= 100

            # Small bonus for fuel efficiency
            cost -= fuel_remaining * 0.001

            return max(0, cost)

        except Exception as e:
            print(f"Error evaluating solution: {e}")
            return 10000.0  # Penalty for evaluation errors

    def get_neighbor(self, solution: List[float]) -> List[float]:
        """Generate a neighboring solution by making small modifications"""
        neighbor = copy.deepcopy(solution)
        
        # Choose how many positions to modify (1-3)
        num_modifications = random.randint(1, 3)
        
        for _ in range(num_modifications):
            pos = random.randint(0, len(neighbor) - 1)
            
            # Different types of modifications
            modification_type = random.random()
            
            if modification_type < 0.3:
                # Turn thrust on/off
                if neighbor[pos] == 0:
                    neighbor[pos] = random.uniform(8.0, 200.0)
                else:
                    neighbor[pos] = 0.0
            elif modification_type < 0.7:
                # Small adjustment if thrust is on
                if neighbor[pos] > 0:
                    adjustment = random.gauss(0, 15)
                    neighbor[pos] += adjustment
                    neighbor[pos] = max(0, min(200, neighbor[pos]))
                    if 0 < neighbor[pos] < 8:
                        neighbor[pos] = 8.0
                else:
                    # Turn on thrust if it was off
                    neighbor[pos] = random.uniform(8.0, 200.0)
            else:
                # Large adjustment or random replacement
                if random.random() < 0.5:
                    neighbor[pos] = 0.0
                else:
                    neighbor[pos] = random.uniform(8.0, 200.0)
        
        return neighbor

    def acceptance_probability(self, old_cost: float, new_cost: float, temperature: float) -> float:
        """Calculate probability of accepting a worse solution"""
        if new_cost < old_cost:
            return 1.0
        if temperature <= 0:
            return 0.0
        try:
            return math.exp(-(new_cost - old_cost) / temperature)
        except OverflowError:
            return 0.0

    def optimize(self, max_iterations=1000) -> Tuple[List[float], float]:
        """Run simulated annealing optimization using real lunar executable"""
        print("Using real lunar executable for cost evaluation...")
        print("This will be slower but more accurate!")

        # Initialize with random solution
        current_solution = self.create_random_solution()
        current_cost = self.cost_function(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        # Validate initial state
        self._validate_solution_cost_sync(best_solution, best_cost, "INITIAL")
        
        temperature = self.initial_temp
        iteration = 0
        
        print(f"Initial solution cost: {current_cost:.2f}")
        
        while temperature > self.min_temp and iteration < max_iterations:
            iteration += 1
            
            # Generate neighbor solution
            neighbor_solution = self.get_neighbor(current_solution)
            neighbor_cost = self.cost_function(neighbor_solution)
            
            # Decide whether to accept the neighbor
            if self.acceptance_probability(current_cost, neighbor_cost, temperature) > random.random():
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                
                # Update best solution if this is better
                if current_cost < best_cost:
                    best_solution = copy.deepcopy(current_solution)
                    best_cost = current_cost
                    
                    # Validate that best_solution and best_cost are in sync
                    self._validate_solution_cost_sync(best_solution, best_cost, f"ITER_{iteration}")
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Print progress every 50 iterations
            if iteration % 50 == 0:
                # Always re-evaluate best solution for progress reporting
                impact, fuel, crashed = self.executor.simulate(best_solution)
                status = "CRASHED" if crashed else "LANDED"
                print(f"Iteration {iteration}: T={temperature:.2f}, "
                      f"Current={current_cost:.2f}, Best={best_cost:.2f}")
                print(f"  Best solution: {impact:.2f} mph, {fuel:.0f} lbs fuel, {status}")
                
                # Validate during progress reporting too
                self._validate_solution_cost_sync(best_solution, best_cost, f"PROGRESS_{iteration}")
        
        # Final validation before returning
        self._validate_solution_cost_sync(best_solution, best_cost, "FINAL")
        
        print(f"Optimization completed after {iteration} iterations")
        return best_solution, best_cost

    def _validate_solution_cost_sync(self, solution: List[float], expected_cost: float, label: str):
        """Validate that a solution's actual cost matches the expected cost"""
        actual_cost = self.cost_function(solution)
        if abs(actual_cost - expected_cost) > 0.01:  # Allow small floating point errors
            print(f"WARNING {label}: Solution/cost out of sync!")
            print(f"  Expected cost: {expected_cost:.2f}")
            print(f"  Actual cost: {actual_cost:.2f}")
            print(f"  Difference: {abs(actual_cost - expected_cost):.2f}")
            
            # Get the raw simulation results for debugging
            impact, fuel, crashed = self.executor.simulate(solution)
            print(f"  Solution evaluates to: {impact:.2f} mph, {fuel:.0f} lbs, {'CRASHED' if crashed else 'LANDED'}")
            
            # This is a serious bug that should be investigated
            raise RuntimeError(f"Solution and cost tracking are out of sync at {label}")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Simulated Annealing optimizer for the Lunar Lander game",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-t", "--initial-temp",
        type=float,
        default=1000.0,
        help="Initial temperature for simulated annealing"
    )

    parser.add_argument(
        "-c", "--cooling-rate",
        type=float,
        default=0.95,
        help="Cooling rate for simulated annealing (0.0 to 1.0)"
    )

    parser.add_argument(
        "-m", "--min-temp",
        type=float,
        default=0.1,
        help="Minimum temperature (stopping criterion)"
    )

    parser.add_argument(
        "-i", "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of iterations"
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
        default="optimized_input_sa.txt",
        help="Output filename for the optimized fuel sequence"
    )

    parser.add_argument(
        "-s", "--sequence-length",
        type=int,
        default=25,
        help="Length of the fuel rate sequence"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.initial_temp <= 0:
        parser.error("Initial temperature must be positive")

    if not 0.0 < args.cooling_rate < 1.0:
        parser.error("Cooling rate must be between 0.0 and 1.0")

    if args.min_temp <= 0:
        parser.error("Minimum temperature must be positive")

    if args.max_iterations < 1:
        parser.error("Maximum iterations must be at least 1")

    if not os.path.exists(args.lunar_path):
        parser.error(f"Lunar executable not found at: {args.lunar_path}")

    if args.sequence_length < 1:
        parser.error("Sequence length must be at least 1")

    return args


def main():
    """Run the optimization using real lunar executable"""
    args = parse_arguments()

    print("Lunar Lander Simulated Annealing Optimizer (Real Executable)")
    print("=" * 60)
    print(f"Parameters:")
    print(f"  Initial temperature: {args.initial_temp}")
    print(f"  Cooling rate: {args.cooling_rate}")
    print(f"  Minimum temperature: {args.min_temp}")
    print(f"  Maximum iterations: {args.max_iterations}")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Lunar executable: {args.lunar_path}")
    print(f"  Output file: {args.output}")
    print()

    optimizer = SimulatedAnnealingOptimizer(
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        min_temp=args.min_temp
    )
    optimizer.executor.lunar_path = args.lunar_path
    optimizer.chromosome_length = args.sequence_length

    best_solution, optimization_best_cost = optimizer.optimize(
        max_iterations=args.max_iterations)

    print("\nOptimization Complete!")
    print("=" * 30)

    # ALWAYS do fresh final evaluation - this is the authoritative result
    executor = LunarExecutor(args.lunar_path)
    
    # Final authoritative evaluation (always trust this over optimization tracking)
    impact_velocity, fuel_remaining, crashed = executor.simulate(best_solution)
    
    # Validate that optimization tracking was correct
    final_cost = optimizer.cost_function(best_solution)
    if abs(final_cost - optimization_best_cost) > 0.01:
        print("WARNING: Optimization cost tracking was incorrect!")
        print(f"  Tracked cost: {optimization_best_cost:.2f}")
        print(f"  Actual cost: {final_cost:.2f}")
        print("  Using final evaluation as authoritative result.")

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
    
    # Verify the saved file gives consistent results
    test_result = subprocess.run(
        ["./lunar", "--echo"],
        stdin=open(args.output, 'r'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30
    )
    
    test_impact, test_fuel, test_crashed = executor._parse_output(test_result.stdout)
    if abs(test_impact - impact_velocity) > 1.0:
        print(f"WARNING: File test shows different result!")
        print(f"  Optimizer: {impact_velocity:.2f} mph")
        print(f"  File test: {test_impact:.2f} mph")

    print(f"\nGenerated '{args.output}' for testing with: "
          f"./lunar < {args.output}")


if __name__ == "__main__":
    main()