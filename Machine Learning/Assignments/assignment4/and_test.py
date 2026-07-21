# Import the Neural Network class from network.py
from network import Network


# Training Dataset (AND Gate)
# ================================

# Input combinations for a 2-input AND gate
AND_input = [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]

AND_ideal = [
    [0],
    [0],
    [0],
    [1]
]


# Create the Neural Network
# ================================

network = Network(
    2,      # Number of input neurons
    1,      # Number of hidden neurons
    1,      # Number of output neurons
    0.7,    # Learning rate
    0.9     # Momentum
)


# ================================
# Training Phase
# ================================

# Train the network for 1000 epochs
# One epoch = one complete pass through the entire dataset
for epoch in range(1000):

    # Loop through each training example
    for x, y in zip(AND_input, AND_ideal):

    
        # Step 1: Forward Propagation
        # ----------------------------
        # Compute the network prediction
        network.compute_outputs(x)

        # Step 2: Backpropagation
        # ----------------------------
        # Compute prediction error and gradients
        network.calc_error(y)

        # Step 3: Gradient Descent
        # ----------------------------
        # Update weights and biases
        network.learn()


# Testing (Recall Phase)
# ================================

print("Recall")

# Pass every possible input through the trained network
for x in AND_input:

    # Compute the network's prediction
    prediction = network.compute_outputs(x)[0]

    # Display the input and predicted output
    print(x, "->", round(prediction, 4))