from abc import ABC, abstractmethod

"""
This acts as a 'blueprint' for the other controllers.
The controllers should be designed with respect to these methods.
"""
class BaseController(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        """Initializes the controller.

        :param kwargs: Keyword arguments.
        """
        pass

    @abstractmethod
    def update(self, error, dt):
        """
        Calculates the control signal based on the error.
        
        :param error: Difference between the result and target.
        :param dt: Difference in time.
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Resets the internal state (like integral history) before a new run."""
        pass

    @abstractmethod
    def get_params(self):
        """Returns the trainable parameters (k_p, k_i, k_d etc...)."""
        pass

    @abstractmethod
    def update_params(self, new_params):
        """
        Updates the paramters after JAX has calculated the gradients.
        
        :param new_params: The new parameters.
        """
        pass