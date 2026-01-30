import jax.numpy as jnp
from plants.Base_plant import BasePlant

class DronePlant(BasePlant):

    def __init__(self, mass, init_height, target_height):
        self.mass = mass
        self.h0 = init_height
        self.target = target_height
        self.g = 9.81

        # State: [Velocity, Height]
        self.initial_state = jnp.array([0.0, self.h0])

    def reset(self):
        return self.initial_state

    def get_initial_state(self):
        return self.initial_state

    def get_output(self, state):
        """We only care about the height."""
        return state[1]

    def update(self, state, control_signal, disturbance):
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
        return self.target