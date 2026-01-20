"""
Lab 5: Molecular Dynamics Simulation
Student: Kazakova Victoria, Group 34-9
Course: Computer Modeling

This simulation models particles in a 2D box with Lennard-Jones potential.
The particles interact through collision and potential forces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class MolecularDynamics:
    """Molecular Dynamics simulation class"""
    
    # Constants
    MIN_DISTANCE = 0.3  # Minimum distance to avoid singularities in force calculation
    
    def __init__(self, n_particles=20, box_size=10.0, dt=0.001):
        """
        Initialize the simulation
        
        Parameters:
        - n_particles: number of particles
        - box_size: size of the simulation box
        - dt: time step
        """
        self.n_particles = n_particles
        self.box_size = box_size
        self.dt = dt
        
        # Initialize random positions (spread out to avoid overlap)
        self.positions = np.random.uniform(2.0, box_size - 2.0, (n_particles, 2))
        
        # Initialize small random velocities
        self.velocities = np.random.randn(n_particles, 2) * 0.05
        
        # Particle properties
        self.masses = np.ones(n_particles)
        self.radius = 0.3
        
        # Lennard-Jones parameters (reduced for stability)
        self.epsilon = 0.1  # depth of potential well
        self.sigma = 0.5    # distance at which potential is zero
        
        # Energy tracking
        self.kinetic_energy_history = []
        self.potential_energy_history = []
        self.total_energy_history = []
        self.time_history = []
        
    def lennard_jones_force(self, r_vec):
        """
        Calculate Lennard-Jones force between two particles
        
        Parameters:
        - r_vec: distance vector between particles
        
        Returns:
        - force vector
        """
        r = np.linalg.norm(r_vec)
        if r < self.MIN_DISTANCE:  # avoid singularity
            r = self.MIN_DISTANCE
        
        # Cutoff distance to limit force range
        r_cutoff = 3.0 * self.sigma
        if r > r_cutoff:
            return np.zeros(2)
            
        # Force magnitude from Lennard-Jones potential
        force_magnitude = 24 * self.epsilon * (
            2 * (self.sigma / r)**13 - (self.sigma / r)**7
        ) / r
        
        force = force_magnitude * r_vec / r
        return force
    
    def calculate_forces(self):
        """Calculate forces on all particles"""
        forces = np.zeros((self.n_particles, 2))
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r_vec = self.positions[j] - self.positions[i]
                
                # Apply periodic boundary conditions
                r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size)
                
                force = self.lennard_jones_force(r_vec)
                forces[i] += force
                forces[j] -= force
                
        return forces
    
    def apply_boundary_conditions(self):
        """Apply periodic boundary conditions"""
        self.positions = self.positions % self.box_size
    
    def calculate_kinetic_energy(self):
        """Calculate total kinetic energy"""
        return 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)
    
    def calculate_potential_energy(self):
        """Calculate total potential energy"""
        potential = 0.0
        
        for i in range(self.n_particles):
            for j in range(i + 1, self.n_particles):
                r_vec = self.positions[j] - self.positions[i]
                r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size)
                r = np.linalg.norm(r_vec)
                
                if r < self.MIN_DISTANCE:
                    r = self.MIN_DISTANCE
                
                # Cutoff distance
                r_cutoff = 3.0 * self.sigma
                if r > r_cutoff:
                    continue
                
                # Lennard-Jones potential
                potential += 4 * self.epsilon * (
                    (self.sigma / r)**12 - (self.sigma / r)**6
                )
        
        return potential
    
    def step(self):
        """Perform one time step using velocity Verlet algorithm"""
        # Calculate forces
        forces = self.calculate_forces()
        
        # Update positions
        self.positions += self.velocities * self.dt + 0.5 * forces / self.masses[:, np.newaxis] * self.dt**2
        
        # Calculate new forces
        new_forces = self.calculate_forces()
        
        # Update velocities
        self.velocities += 0.5 * (forces + new_forces) / self.masses[:, np.newaxis] * self.dt
        
        # Apply boundary conditions
        self.apply_boundary_conditions()
    
    def run_simulation(self, n_steps=5000):
        """Run the simulation for n_steps"""
        for step in range(n_steps):
            self.step()
            
            # Record energies every 50 steps
            if step % 50 == 0:
                ke = self.calculate_kinetic_energy()
                pe = self.calculate_potential_energy()
                self.kinetic_energy_history.append(ke)
                self.potential_energy_history.append(pe)
                self.total_energy_history.append(ke + pe)
                self.time_history.append(step * self.dt)
    
    def plot_energy(self):
        """Plot energy conservation"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_history, self.kinetic_energy_history, label='Kinetic Energy', alpha=0.7)
        plt.plot(self.time_history, self.potential_energy_history, label='Potential Energy', alpha=0.7)
        plt.plot(self.time_history, self.total_energy_history, label='Total Energy', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Energy')
        plt.title('Energy Conservation in Molecular Dynamics')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('energy_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Energy plot saved as 'energy_plot.png'")
    
    def plot_snapshot(self):
        """Plot current particle positions"""
        plt.figure(figsize=(8, 8))
        plt.scatter(self.positions[:, 0], self.positions[:, 1], s=100, alpha=0.6)
        plt.xlim(0, self.box_size)
        plt.ylim(0, self.box_size)
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Particle Positions in Molecular Dynamics Simulation')
        plt.grid(True, alpha=0.3)
        plt.gca().set_aspect('equal')
        plt.savefig('particles_snapshot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Snapshot saved as 'particles_snapshot.png'")


def main():
    """Main function to run the simulation"""
    print("=" * 60)
    print("Lab 5: Molecular Dynamics Simulation")
    print("Student: Kazakova Victoria, Group 34-9")
    print("=" * 60)
    
    # Create simulation
    print("\nInitializing simulation with 20 particles...")
    sim = MolecularDynamics(n_particles=20, box_size=10.0, dt=0.001)
    
    # Run simulation
    print("Running simulation for 5000 steps...")
    sim.run_simulation(n_steps=5000)
    
    # Calculate statistics
    ke_mean = np.mean(sim.kinetic_energy_history)
    pe_mean = np.mean(sim.potential_energy_history)
    total_mean = np.mean(sim.total_energy_history)
    energy_drift = (sim.total_energy_history[-1] - sim.total_energy_history[0]) / sim.total_energy_history[0] * 100
    
    print("\n" + "=" * 60)
    print("Simulation Results:")
    print("=" * 60)
    print(f"Average Kinetic Energy: {ke_mean:.4f}")
    print(f"Average Potential Energy: {pe_mean:.4f}")
    print(f"Average Total Energy: {total_mean:.4f}")
    print(f"Energy Drift: {energy_drift:.4f}%")
    print("=" * 60)
    
    # Generate plots
    print("\nGenerating visualization plots...")
    sim.plot_energy()
    sim.plot_snapshot()
    
    print("\n✓ Simulation completed successfully!")
    print("Check the current folder for output plots (energy_plot.png, particles_snapshot.png)")


if __name__ == "__main__":
    main()
