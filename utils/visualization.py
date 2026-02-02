import matplotlib.pyplot as plt
import numpy as np

"""
This module contains the functions for visualizing the results of the simulation.
"""
def plot_results(mse_history, param_history=None):
    """
    Plots the learning progression and parameter evolution.

    :param mse_history: List of MSE values per epoch.
    :param param_history: Optional list/array of PID parameters per epoch.
    :returns: None
    """
    num_plots = 2 if param_history is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(12, 5))
    
    if num_plots == 1:
        axes = [axes]

    # Plot 1: Learning Progression (MSE)
    ax1 = axes[0]
    ax1.plot(mse_history, label='MSE', color='black')
    ax1.set_title('Learning Progression')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error (MSE)')
    ax1.grid(True)
    
    # Plot 2: Control Parameters (Only for PID)
    if param_history is not None:
        ax2 = axes[1]
        params = np.array(param_history)
        
        ax2.plot(params[:, 0], label='Kp')
        ax2.plot(params[:, 1], label='Ki')
        ax2.plot(params[:, 2], label='Kd')
        
        ax2.set_title('Control Parameters Evolution')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Parameter Value')
        ax2.legend()
        ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_system_response(state_history, plant_name, target=None):
    """
    Plots the system response for each plant.

    :param state_history: History of states.
    :param plant_name: Name of the plant.
    :param target: Optional target value.
    :returns: None
    """
    history = np.array(state_history)
    timesteps = np.arange(len(history))
    
    plt.figure(figsize=(10, 6))
    
    # Bathtub
    if plant_name == "Bathtub":
        plt.plot(timesteps, history, label='Water Level (H)', color='blue')
        plt.ylabel("Height (m)")
        
    # Cournot
    elif plant_name == "Cournot":
        plt.plot(timesteps, history[:, 0], label='Agent (q1)', color='blue')
        plt.plot(timesteps, history[:, 1], label='Competitor (q2)', color='orange', linestyle=':')
        plt.ylabel("Production Quantity")

    # Drone
    elif plant_name == "Drone":
        velocity = history[:, 0]
        height = history[:, 1]
        
        plt.plot(timesteps, height, label='Drone Height', color='blue')
        
        plt.ylabel("Height (m)")

    # Draw target line if it exists
    if target is not None:
        plt.axhline(y=target, color='r', linestyle='--', label='Target')

    plt.title(f"System Response: {plant_name}")
    plt.xlabel("Timestep")
    plt.legend()
    plt.grid(True)
    plt.show()