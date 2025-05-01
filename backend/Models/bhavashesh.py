import random
import numpy as np
import time


def get_random_number(min_val=0, max_val=100):
    """
    Generate a random integer between min_val and max_val (inclusive)
    
    Args:
        min_val (int): Minimum value (default: 0)
        max_val (int): Maximum value (default: 100)
    
    Returns:
        int: A random number between min_val and max_val
    """
    return random.randint(min_val, max_val)


def get_random_float(min_val=0.0, max_val=1.0):
    """
    Generate a random float between min_val and max_val
    
    Args:
        min_val (float): Minimum value (default: 0.0)
        max_val (float): Maximum value (default: 1.0)
    
    Returns:
        float: A random float between min_val and max_val
    """
    return random.uniform(min_val, max_val)


def get_random_gaussian(mean=0.0, std=1.0):
    """
    Generate a random number from a Gaussian/normal distribution
    
    Args:
        mean (float): Mean of the distribution (default: 0.0)
        std (float): Standard deviation of the distribution (default: 1.0)
    
    Returns:
        float: A random number from the specified normal distribution
    """
    return random.gauss(mean, std)


def get_random_seed():
    """
    Get a random seed based on current time
    
    Returns:
        int: A random seed value
    """
    return int(time.time() * 1000) % 10000


def set_random_seed(seed=None):
    """
    Set the random seed for reproducibility
    
    Args:
        seed (int, optional): Seed value to use. If None, uses time-based seed
    """
    if seed is None:
        seed = get_random_seed()
    
    random.seed(seed)
    np.random.seed(seed)
    return seed


if __name__ == "__main__":
    # Example usage
    print(f"Random integer: {get_random_number(1, 100)}")
    print(f"Random float: {get_random_float(0.5, 10.5)}")
    print(f"Random gaussian: {get_random_gaussian(5.0, 2.0)}")
    
    # Set seed for reproducibility
    used_seed = set_random_seed(42)
    print(f"Using seed: {used_seed}")
    
    # Generate same sequence with fixed seed
    print("Random sequence with fixed seed:")
    for _ in range(5):
        print(get_random_number(1, 10), end=" ")