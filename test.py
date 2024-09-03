import matplotlib.pyplot as plt
import numpy as np
import chain

def calculate_mse(exact_prob, sampled_prob):
    return np.mean((exact_prob - sampled_prob) ** 2)  # Mean Squared Error

def generate_synthetic_chain(N, K):
    # Create a synthetic chain with random potential functions
    c = chain.Chain(N, K)
    return c

def main():
    # Generate synthetic chain
    N = 10  # Number of variables in the chain
    K = 10  # Degree of each variable
    c = generate_synthetic_chain(N, K)

    # Exact probabilities
    exact_prob = c.exact()

    # Initialize lists to store results
    steps_list = []
    mse_list = []  # Mean Squared Error

    # Experiment with varying number of steps
    max_steps = 1000
    for steps in range(1, max_steps + 1):
        # Sampled probabilities
        sampled_prob = c.sample(steps)

        # Calculate MSE
        mse = calculate_mse(exact_prob, sampled_prob)

        # Store results
        steps_list.append(steps)
        mse_list.append(mse)

    # Plotting
    plt.plot(steps_list, mse_list)
    plt.xlabel('Number of Steps')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Convergence of Gibbs Sampler')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
