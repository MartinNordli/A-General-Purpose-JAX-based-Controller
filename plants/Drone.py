import jax.numpy as jnp
from plants.Base_plant import BasePlant

"""
The system simulates a drone flying in the air.
"""
class DronePlant(BasePlant):

    def __init__(self, mass, init_height, target_height):
        """Initializes the drone plant.

        :param mass: Mass of the drone.
        :param init_height: Initial height of the drone.
        :param target_height: Target height of the drone.
        """
        self.mass = mass
        self.h0 = init_height
        self.target = target_height
        self.g = 9.81

        # State: [Velocity, Height]
        self.initial_state = jnp.array([0.0, self.h0])

    def reset(self):
        """Resets the internal state (used if running outside JAX scan)."""
        return self.initial_state

    def get_initial_state(self):
        """Returns the initial state of the plant."""
        return self.initial_state

    def get_output(self, state):
        """We only care about the height."""
        return state[1]

    def update(self, state, control_signal, disturbance):
        """Updates the state of the plant.

        :param state: Current state of the plant.
        :param control_signal: Control signal from the controller.
        :param disturbance: Disturbance from the environment.

        :returns: New state of the plant.
        """
        velocity, height = state
        U = control_signal * 100.0
        D = disturbance
        dt = 0.1 # Timestep

        # Constraint: The motor cannot yield a negative force and has a maximum force of 50.0.
        U_clamped = jnp.clip(U, 0.0, 50.0)

        # Newtons 2. law: F = ma  ->  a = F/m
        # Forces: Thrust (U) + wind (D) - weight (m*g)
        force = U_clamped + D - (self.mass * self.g)

        acceleration = force / self.mass

        new_velocity = velocity + acceleration * dt

        new_height = height + new_velocity * dt

        # The drones height can´t be below the ground.
        new_height = jnp.maximum(new_height, 0.0)

        # If the drone falls to the ground, the velocity should be set to zero.
        new_velocity = jnp.where(new_height == 0.0, 0.0, new_velocity)

        return jnp.stack([new_velocity, new_height])


    def get_target(self):
        """Returns the target height of the plant."""
        return self.target