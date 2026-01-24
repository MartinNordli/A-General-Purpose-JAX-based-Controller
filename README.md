Her er et forslag til en profesjonell `README.md`. Den er strukturert for å vise at du har forstått både teknologien (JAX) og designprinsippene (Modulæritet/"The Critical Divide").

Du kan kopiere hele teksten nedenfor inn i en fil som heter `README.md` i rotmappen din.

---

# General Purpose JAX Controller

**Course:** IT-3105 AI Programming

**Language:** Python 3 (JAX)

## Project Overview

This project implements a general-purpose control system framework capable of regulating various simulated environments ("Plants") using both classical and AI-driven control strategies.

The core of the system is built on **JAX**, utilizing its automatic differentiation capabilities to optimize controller parameters via Gradient Descent. By tracing the execution path through the simulation loop using `jax.lax.scan`, the system can compute gradients for the entire episode history, enabling efficient learning for both PID parameters and Neural Network weights.

## Design Philosophy: "The Critical Divide"

Strict adherence to the **Separation of Concerns** principle is central to this implementation. The system is architected to maintain a clean boundary between the **AI/Controller** and the **Simulated World (Plant)**.

* **The Controller** acts as a general-purpose agent. It perceives an *error* signal and outputs a *control signal* (U), without inherent knowledge of the physical laws governing the plant.
* **The Plant** encapsulates the domain logic, physics, and state transitions. It exposes a standardized interface (`update`, `reset`, `get_target`) but remains passive regarding control logic.
* **CONSYS** serves as the integration layer, managing the feedback loop and facilitating data flow between the two components.

## File Structure

```text
├── main.py                  # Entry point: Orchestrates training and visualization
├── config.py                # Central configuration for hyperparameters and simulation settings
├── consys.py                # System Wrapper: Implements the JAX-optimized simulation loop
│
├── controllers/             # Control Algorithms
│   ├── Base_controller.py   # Abstract base class for all controllers
│   ├── Classic_controller.py# PID implementation (Proportional-Integral-Derivative)
│   └── Neural_network_controller.py
│
├── plants/                  # Simulated Environments
│   ├── Base_plant.py        # Abstract base class for all plants
│   ├── Bathtub.py           # Physics model of a bathtub with fluid dynamics
│   ├── Cournot.py           # Economic simulation of quantity competition
│   └── Plant3.py            # Custom domain implementation
│
└── utils/
    └── visualization.py     # Plotting tools for MSE and parameter evolution

```

## Getting Started

### Prerequisites

* Python 3.8+
* JAX
* NumPy
* Matplotlib

To install the necessary dependencies, run:

```bash
pip install jax jaxlib numpy matplotlib

```

*(Note: For JAX with GPU support, please refer to the [official JAX installation guide](https://www.google.com/search?q=https://github.com/google/jax%23installation).)*

### Running the Simulation

The system is configured via `config.py`. To execute a training run:

1. Open `config.py` and select your desired Plant and Controller:
```python
PLANT_TO_RUN = "Bathtub"      # Options: "Bathtub", "Cournot"
CONTROLLER_TO_USE = "Classic" # Options: "Classic", "NeuralNet"

```


2. Run the main script:
```bash
python main.py

```


3. The system will output the MSE (Mean Squared Error) for every 10th epoch in the terminal. Upon completion, it will generate plots visualizing the learning progression and parameter evolution.

## Configuration

All pivotal parameters are centralized in `config.py` to facilitate easy experimentation:

* **Simulation:** `NUM_EPOCHS`, `TIMESTEPS_PER_EPOCH`, `LEARNING_RATE`, `DISTURBANCE`.
* **PID Settings:** Initial `Kp`, `Ki`, `Kd` values.
* **Plant Physics:**
* *Bathtub:* Area, Drain size, Initial height.
* *Cournot:* Max price, Marginal cost.



## Visualization

The project includes a dedicated visualization module that produces:

1. **Learning Progression:** A plot of the Mean Squared Error (MSE) over epochs, demonstrating the system's ability to minimize error.
2. **Parameter Evolution:** (For PID) A plot showing how `Kp`, `Ki`, and `Kd` adapt over time to optimize control.