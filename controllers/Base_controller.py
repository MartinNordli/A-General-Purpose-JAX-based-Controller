from abc import ABC, abstractmethod

class BaseController(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, error, dt):
        """
        Calculates the control signal based on the error.
        
        :param error: Description
        :param dt: Description
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Resets the internal state (like integral history) before a new run.
        """
        pass

    @abstractmethod
    def get_params(self):
        """
        Returns the trainable parameters (k_p, k_i, k_d etc...).
        """
        pass

    @abstractmethod
    def update_params(self, new_params):
        """
        Updates the paramters after JAX has calculated the gradients.
        
        :param new_params: The new parameters.
        """
        pass