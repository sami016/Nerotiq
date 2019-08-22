double activation(double input) {
    if (input < 0.0) {
        return 0.0;
    }
    return input;
}

double activationDerivative(double input) {
    if (input <= 0.0) {
        return 0.0;
    }
    return 1;
}