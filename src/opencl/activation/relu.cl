float activation(float input) {
    if (input < 0.0) {
        return 0.0;
    }
    return input;
}

float activationDerivative(float input) {
    if (input <= 0.0) {
        return 0.0;
    }
    return 1;
}