"""
Lab 6: Monte Carlo Simulation and Random Processes
Student: Kazakova Victoria, Group 34-9
Course: Computer Modeling

This simulation includes:
1. Random Walk in 2D
2. Monte Carlo estimation of Pi
3. Statistical analysis of random processes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


class RandomWalk:
    """2D Random Walk simulation"""
    
    def __init__(self, n_steps=1000, step_size=1.0):
        """
        Initialize random walk
        
        Parameters:
        - n_steps: number of steps
        - step_size: size of each step
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.positions = np.zeros((n_steps + 1, 2))
        
    def simulate(self):
        """Perform the random walk"""
        for i in range(self.n_steps):
            # Random angle
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Step in random direction
            dx = self.step_size * np.cos(angle)
            dy = self.step_size * np.sin(angle)
            
            self.positions[i + 1] = self.positions[i] + np.array([dx, dy])
    
    def plot_walk(self):
        """Plot the random walk trajectory"""
        plt.figure(figsize=(10, 10))
        
        # Plot trajectory
        plt.plot(self.positions[:, 0], self.positions[:, 1], 
                alpha=0.6, linewidth=0.5, color='blue')
        
        # Mark start and end
        plt.scatter(self.positions[0, 0], self.positions[0, 1], 
                   c='green', s=200, marker='o', label='Start', zorder=5)
        plt.scatter(self.positions[-1, 0], self.positions[-1, 1], 
                   c='red', s=200, marker='x', label='End', zorder=5)
        
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title(f'2D Random Walk ({self.n_steps} steps)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.savefig('random_walk.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Random walk plot saved as 'random_walk.png'")
    
    def calculate_displacement(self):
        """Calculate displacement statistics"""
        displacements = np.linalg.norm(self.positions, axis=1)
        return displacements


class MonteCarloPi:
    """Monte Carlo estimation of Pi"""
    
    def __init__(self, n_samples=10000):
        """
        Initialize Monte Carlo simulation
        
        Parameters:
        - n_samples: number of random samples
        """
        self.n_samples = n_samples
        self.points = None
        self.inside_circle = None
        
    def simulate(self):
        """Perform Monte Carlo simulation"""
        # Generate random points in [0, 1] x [0, 1]
        self.points = np.random.uniform(0, 1, (self.n_samples, 2))
        
        # Check if points are inside quarter circle
        distances = np.linalg.norm(self.points, axis=1)
        self.inside_circle = distances <= 1.0
        
    def estimate_pi(self):
        """Estimate Pi from the simulation"""
        n_inside = np.sum(self.inside_circle)
        pi_estimate = 4.0 * n_inside / self.n_samples
        return pi_estimate
    
    def plot_points(self):
        """Plot the Monte Carlo points"""
        plt.figure(figsize=(8, 8))
        
        # Plot points
        plt.scatter(self.points[self.inside_circle, 0], 
                   self.points[self.inside_circle, 1],
                   c='blue', s=1, alpha=0.5, label='Inside circle')
        plt.scatter(self.points[~self.inside_circle, 0], 
                   self.points[~self.inside_circle, 1],
                   c='red', s=1, alpha=0.5, label='Outside circle')
        
        # Draw quarter circle
        theta = np.linspace(0, np.pi/2, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2, label='Quarter circle')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Monte Carlo Estimation of π (n={self.n_samples})')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        plt.savefig('monte_carlo_pi.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Monte Carlo Pi plot saved as 'monte_carlo_pi.png'")


class MultipleRandomWalks:
    """Multiple random walks for statistical analysis"""
    
    def __init__(self, n_walks=100, n_steps=500):
        """
        Initialize multiple random walks
        
        Parameters:
        - n_walks: number of independent walks
        - n_steps: number of steps per walk
        """
        self.n_walks = n_walks
        self.n_steps = n_steps
        self.final_distances = []
        
    def simulate(self):
        """Simulate multiple random walks"""
        for _ in range(self.n_walks):
            walk = RandomWalk(n_steps=self.n_steps)
            walk.simulate()
            
            # Calculate final distance from origin
            final_distance = np.linalg.norm(walk.positions[-1])
            self.final_distances.append(final_distance)
        
        self.final_distances = np.array(self.final_distances)
    
    def plot_distribution(self):
        """Plot distribution of final distances"""
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.hist(self.final_distances, bins=30, density=True, 
                alpha=0.7, edgecolor='black', label='Observed')
        
        # Theoretical distribution (Rayleigh for 2D random walk)
        sigma = np.sqrt(self.n_steps / 2)
        x = np.linspace(0, max(self.final_distances), 100)
        rayleigh_pdf = stats.rayleigh.pdf(x, scale=sigma)
        plt.plot(x, rayleigh_pdf, 'r-', linewidth=2, label='Rayleigh (theoretical)')
        
        plt.xlabel('Final Distance from Origin')
        plt.ylabel('Probability Density')
        plt.title(f'Distribution of Final Distances ({self.n_walks} walks, {self.n_steps} steps each)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('distance_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Distance distribution plot saved as 'distance_distribution.png'")


def convergence_study():
    """Study convergence of Monte Carlo Pi estimation"""
    sample_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    pi_estimates = []
    errors = []
    
    print("\nConvergence Study for Monte Carlo Pi Estimation:")
    print("-" * 60)
    
    for n in sample_sizes:
        mc = MonteCarloPi(n_samples=n)
        mc.simulate()
        pi_est = mc.estimate_pi()
        error = abs(pi_est - np.pi)
        
        pi_estimates.append(pi_est)
        errors.append(error)
        
        print(f"n = {n:>7}: π ≈ {pi_est:.6f}, error = {error:.6f}")
    
    print("-" * 60)
    
    # Plot convergence
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(sample_sizes, pi_estimates, 'bo-', linewidth=2, markersize=8)
    plt.axhline(y=np.pi, color='r', linestyle='--', linewidth=2, label='True π')
    plt.xlabel('Number of Samples')
    plt.ylabel('Estimated π')
    plt.title('Convergence of Monte Carlo Pi Estimation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.loglog(sample_sizes, errors, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Samples')
    plt.ylabel('Absolute Error')
    plt.title('Error vs Sample Size')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Convergence study plot saved as 'convergence_study.png'")


def main():
    """Main function to run all simulations"""
    print("=" * 60)
    print("Lab 6: Monte Carlo Simulation and Random Processes")
    print("Student: Kazakova Victoria, Group 34-9")
    print("=" * 60)
    
    # Part 1: Random Walk
    print("\n[1/4] Simulating 2D Random Walk...")
    walk = RandomWalk(n_steps=5000, step_size=1.0)
    walk.simulate()
    walk.plot_walk()
    
    final_distance = np.linalg.norm(walk.positions[-1])
    print(f"Final distance from origin: {final_distance:.2f}")
    
    # Part 2: Monte Carlo Pi Estimation
    print("\n[2/4] Monte Carlo Estimation of π...")
    mc = MonteCarloPi(n_samples=50000)
    mc.simulate()
    pi_estimate = mc.estimate_pi()
    mc.plot_points()
    
    print(f"Estimated π = {pi_estimate:.6f}")
    print(f"True π      = {np.pi:.6f}")
    print(f"Error       = {abs(pi_estimate - np.pi):.6f}")
    
    # Part 3: Multiple Random Walks
    print("\n[3/4] Simulating multiple random walks...")
    multi_walks = MultipleRandomWalks(n_walks=500, n_steps=500)
    multi_walks.simulate()
    multi_walks.plot_distribution()
    
    mean_distance = np.mean(multi_walks.final_distances)
    std_distance = np.std(multi_walks.final_distances)
    print(f"Mean final distance: {mean_distance:.2f} ± {std_distance:.2f}")
    
    # Part 4: Convergence Study
    print("\n[4/4] Running convergence study...")
    convergence_study()
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    print("✓ 2D Random Walk completed")
    print("✓ Monte Carlo Pi estimation completed")
    print("✓ Statistical analysis of random walks completed")
    print("✓ Convergence study completed")
    print("\nAll plots saved in current folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
