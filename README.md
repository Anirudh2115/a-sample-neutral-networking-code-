#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_NODES 2
#define HIDDEN_NODES 2
#define OUTPUT_NODES 1
#define LEARNING_RATE 0.1
#define EPOCHS 10000

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Structure for the neural network
typedef struct {
    double input[INPUT_NODES];
    double hidden[HIDDEN_NODES];
    double output[OUTPUT_NODES];
    double weights_input_hidden[INPUT_NODES][HIDDEN_NODES];
    double weights_hidden_output[HIDDEN_NODES][OUTPUT_NODES];
    double bias_hidden[HIDDEN_NODES];
    double bias_output[OUTPUT_NODES];
} NeuralNetwork;

// Function to initialize the neural network
void initialize_network(NeuralNetwork *nn) {
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->weights_input_hidden[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        nn->bias_hidden[i] = (double)rand() / RAND_MAX;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->weights_hidden_output[i][j] = (double)rand() / RAND_MAX;
        }
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        nn->bias_output[i] = (double)rand() / RAND_MAX;
    }
}

// Function to train the neural network
void train(NeuralNetwork *nn, double input[][INPUT_NODES], double target[], int samples) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int s = 0; s < samples; s++) {
            // Forward pass
            for (int i = 0; i < HIDDEN_NODES; i++) {
                nn->hidden[i] = nn->bias_hidden[i];
                for (int j = 0; j < INPUT_NODES; j++) {
                    nn->hidden[i] += input[s][j] * nn->weights_input_hidden[j][i];
                }
                nn->hidden[i] = sigmoid(nn->hidden[i]);
            }

            for (int i = 0; i < OUTPUT_NODES; i++) {
                nn->output[i] = nn->bias_output[i];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    nn->output[i] += nn->hidden[j] * nn->weights_hidden_output[j][i];
                }
                nn->output[i] = sigmoid(nn->output[i]);
            }

            // Backward pass
            double output_error[OUTPUT_NODES];
            double hidden_error[HIDDEN_NODES];

            for (int i = 0; i < OUTPUT_NODES; i++) {
                output_error[i] = target[s] - nn->output[i];
                for (int j = 0; j < HIDDEN_NODES; j++) {
                    nn->weights_hidden_output[j][i] += LEARNING_RATE * output_error[i] * sigmoid_derivative(nn->output[i]) * nn->hidden[j];
                }
                nn->bias_output[i] += LEARNING_RATE * output_error[i] * sigmoid_derivative(nn->output[i]);
            }

            for (int i = 0; i < HIDDEN_NODES; i++) {
                hidden_error[i] = 0.0;
                for (int j = 0; j < OUTPUT_NODES; j++) {
                    hidden_error[i] += output_error[j] * nn->weights_hidden_output[i][j];
                }
                for (int j = 0; j < INPUT_NODES; j++) {
                    nn->weights_input_hidden[j][i] += LEARNING_RATE * hidden_error[i] * sigmoid_derivative(nn->hidden[i]) * input[s][j];
                }
                nn->bias_hidden[i] += LEARNING_RATE * hidden_error[i] * sigmoid_derivative(nn->hidden[i]);
            }
        }
    }
}

// Function to predict output for a given input
double predict(NeuralNetwork *nn, double input[INPUT_NODES]) {
    for (int i = 0; i < HIDDEN_NODES; i++) {
        nn->hidden[i] = nn->bias_hidden[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            nn->hidden[i] += input[j] * nn->weights_input_hidden[j][i];
        }
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        nn->output[i] = nn->bias_output[i];
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->output[i] += nn->hidden[j] * nn->weights_hidden_output[j][i];
        }
        nn->output[i] = sigmoid(nn->output[i]);
    }

    return nn->output[0];
}

int main() {
    NeuralNetwork nn;
    initialize_network(&nn);

    // XOR input and target output
    double input[4][INPUT_NODES] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double target[4] = {0, 1, 1, 0};  // XOR target

    train(&nn, input, target, 4);

    // Test the trained model
    printf("Predictions after training:\n");
    for (int i = 0; i < 4; i++) {
        double prediction = predict(&nn, input[i]);
        printf("Input: (%f, %f) => Prediction: %f\n", input[i][0], input[i][1], prediction);
    }

    return 0;
}
