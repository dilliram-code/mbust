import math
import random


class Network:
    """
    A simple Feedforward Neural Network with one hidden layer.
    """

    def __init__(self, input_count, hidden_count, output_count,
                learn_rate=0.7, momentum=0.9):
        """
        Initialize the neural network.
        """

        # Hyperparameters
        self.learn_rate = learn_rate
        self.momentum = momentum

        # Network architecture
        self.input_count = input_count
        self.hidden_count = hidden_count
        self.output_count = output_count

        # Total neurons in the network
        self.neuron_count = input_count + hidden_count + output_count

        # Total weights:
        # Input → Hidden + Hidden → Output
        self.weight_count = (
            (input_count * hidden_count) +
            (hidden_count * output_count)
        )

        # Sum of squared errors for the current epoch
        self.global_error = 0

        # Stores output (activation) of every neuron
        self.fire = [0.0] * self.neuron_count

        # Weight matrix stored as a flattened 1D array
        self.matrix = [0.0] * self.weight_count

        # Previous weight update (used for momentum)
        self.matrix_delta = [0.0] * self.weight_count

        # Accumulated weight gradients before updating
        self.acc_matrix_delta = [0.0] * self.weight_count

        # Bias (threshold) value for every neuron
        self.thresholds = [0.0] * self.neuron_count

        # Previous bias update (momentum)
        self.threshold_delta = [0.0] * self.neuron_count

        # Accumulated bias gradients
        self.acc_threshold_delta = [0.0] * self.neuron_count

        # Error at every neuron
        self.error = [0.0] * self.neuron_count

        # Local gradient (δ) for every neuron
        self.error_delta = [0.0] * self.neuron_count

        # Initialize weights and biases randomly
        self.reset()

    def threshold(self, x):
        """
        Sigmoid activation function.
        """
        return 1 / (1 + math.exp(-x))

    def compute_outputs(self, inputs):
        """
        Perform forward propagation.
        """

        # Starting index of hidden layer
        hidden_index = self.input_count

        # Starting index of output layer
        output_index = self.input_count + self.hidden_count

        # Load input values into input neurons
        for i in range(self.input_count):
            self.fire[i] = inputs[i]

        # Index into the flattened weight matrix
        idx = 0

        # ==========================
        # Forward Pass: Hidden Layer
        # ==========================
        for i in range(hidden_index, output_index):

            # Start with the neuron's bias
            s = self.thresholds[i]

            # Weighted sum of all input neurons
            for j in range(self.input_count):
                s += self.fire[j] * self.matrix[idx]
                idx += 1

            # Apply sigmoid activation
            self.fire[i] = self.threshold(s)

        result = []

        # ==========================
        # Forward Pass: Output Layer
        # ==========================
        for i in range(output_index, self.neuron_count):

            # Start with bias
            s = self.thresholds[i]

            # Weighted sum from hidden neurons
            for j in range(hidden_index, output_index):
                s += self.fire[j] * self.matrix[idx]
                idx += 1

            # Final output activation
            self.fire[i] = self.threshold(s)

            result.append(self.fire[i])

        return result

    def calc_error(self, ideal):
        """
        Perform backpropagation.
        """

        hidden_index = self.input_count
        output_index = self.input_count + self.hidden_count

        # Clear previous errors
        for i in range(self.input_count, self.neuron_count):
            self.error[i] = 0

        # ==========================
        # Output Layer Error
        # ==========================
        for i in range(output_index, self.neuron_count):

            # Desired - Predicted
            self.error[i] = ideal[i - output_index] - self.fire[i]

            # Accumulate squared error
            self.global_error += self.error[i] ** 2

            # δ = error × sigmoid derivative
            self.error_delta[i] = (
                self.error[i] *
                self.fire[i] *
                (1 - self.fire[i])
            )

        # First Hidden→Output weight index
        winx = self.input_count * self.hidden_count

        # =====================================
        # Backpropagate Output → Hidden Layer
        # =====================================
        for i in range(output_index, self.neuron_count):

            for j in range(hidden_index, output_index):

                # Accumulate weight gradient
                self.acc_matrix_delta[winx] += (
                    self.error_delta[i] * self.fire[j]
                )

                # Pass error backward
                self.error[j] += (
                    self.matrix[winx] *
                    self.error_delta[i]
                )

                winx += 1

            # Accumulate bias gradient
            self.acc_threshold_delta[i] += self.error_delta[i]

        # Hidden Layer Gradients
        # ==========================
        for i in range(hidden_index, output_index):

            self.error_delta[i] = (
                self.error[i] *
                self.fire[i] *
                (1 - self.fire[i])
            )

        # Reset to Input→Hidden weights
        winx = 0

        # Backpropagate Hidden → Input Layer
        # =====================================
        for i in range(hidden_index, output_index):

            for j in range(hidden_index):

                # Accumulate gradient
                self.acc_matrix_delta[winx] += (
                    self.error_delta[i] * self.fire[j]
                )

                # Continue propagating error
                self.error[j] += (
                    self.matrix[winx] *
                    self.error_delta[i]
                )

                winx += 1

            # Accumulate bias gradient
            self.acc_threshold_delta[i] += self.error_delta[i]

    def learn(self):
        """
        Update weights and biases.
        """

        # ==========================
        # Update Weights
        # ==========================
        for i in range(len(self.matrix)):

            # Compute weight update
            self.matrix_delta[i] = (
                self.learn_rate *
                self.acc_matrix_delta[i] +
                self.momentum *
                self.matrix_delta[i]
            )

            # Apply update
            self.matrix[i] += self.matrix_delta[i]

            # Clear accumulated gradient
            self.acc_matrix_delta[i] = 0
            
        # Update Biases
        # ==========================
        for i in range(self.input_count, self.neuron_count):

            self.threshold_delta[i] = (
                self.learn_rate *
                self.acc_threshold_delta[i] +
                self.momentum *
                self.threshold_delta[i]
            )

            self.thresholds[i] += self.threshold_delta[i]

            # Reset accumulated bias gradient
            self.acc_threshold_delta[i] = 0

    def get_error(self, length):
        """
        Compute Root Mean Squared Error (RMSE).
        """

        err = math.sqrt(
            self.global_error /
            (length * self.output_count)
        )

        # Reset error accumulator for next epoch
        self.global_error = 0

        return err

    def reset(self):
        """
        Randomly initialize all weights and biases.
        """

        # Initialize neuron biases
        for i in range(self.neuron_count):

            self.thresholds[i] = 0.5 - random.random()

            self.threshold_delta[i] = 0

            self.acc_threshold_delta[i] = 0

        # Initialize connection weights
        for i in range(len(self.matrix)):

            self.matrix[i] = 0.5 - random.random()

            self.matrix_delta[i] = 0

            self.acc_matrix_delta[i] = 0