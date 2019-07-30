/*
 * Standard feed-forward neural network layer implementation.
 */


/**
 * Calculate the activation and output values during the forward pass.
 * These are calculated by:
 * - summing the products of the inputs and weights (plus bias).
 * - applying the activation function (defined externally).
 * 
 * previousLayerCount: number of nodes in the previous layer.
 * layerNodeCount: number of nodes in the current layer.
 * previousLayerOutputs: pointer to previous layer outputs vector.
 * layerWeights: pointer to layer weights matrix.
 * layerBiases: pointer to layer bias vector.
 * layerSums: pointer to layer activations vector.
 * layerOutputs: pointer to layer outputs vector.
 */
__kernel void forwardPass(
    uint previousLayerNodeCount, // 0
    uint layerNodeCount, // 1
    __global float *previousLayerOutputs, // 2
    __global float *layerWeights, // 3
    __global float *layerBiases, // 4
    __global float *layerSums, // 5
    __global float *layerOutputs // 6
)
{
    // Get the current node id we're processing.
    int node_id = get_global_id(0);
    // Sum up the product of the inputs and the weights.
    float sum = layerBiases[node_id];
    for (int i=0; i<previousLayerNodeCount; i++)
    {
        float weight = layerWeights[i * layerNodeCount + node_id];
        float input = previousLayerOutputs[i];
        sum += weight * input;
    }
    // Set the activation to the accumulated sum.
    layerSums[node_id] = sum;
    // Set the output to the activation of the sum.
    layerOutputs[node_id] = activation(sum);
}

/*
 * Standard feed-forward neural network layer implementation.
 */


/**
 * Propagates errors backwards across the layer using delta values.
 * These are calculated by:
 * - Summing the product of the delta values and weights to each node in the next layer.
 * - applying derivative of the activation function (defined externally).
 * 
 * layerNodeCount: number of nodes in the current layer.
 * nextLayerNodeCount: number of nodes in the next layer.
 * layerSums: pointer to layer activations vector.
 * layerOutputs: pointer to layer outputs vector.
 * layerDeltas: pointer to layer delta value vector.
 * nextLayerDeltas: pointer to the next layer's delta value vector. (0 if last layer)
 * nextLayerWeights: pointer to the next layer's weight matrix vector. (0 if last layer)
 * layerBiases: pointer to targets. (0 unless last layer).
 */
__kernel void backwardPass(
    uint previousLayerNodeCount, // 0
    uint layerNodeCount, // 1
    uint nextLayerNodeCount, // 2
    __global float *layerSums, // 3
    __global float *layerOutputs, // 4
    __global float *layerDeltas, // 5
    __global float *layerWeights, // 6
    __global float *layerBiases, // 7
    __global float *previousLayerOutputs, // 8
    __global float *nextLayerDeltas, // 9
    __global float *nextLayerWeights, // 10
    __global float *targets // 11
)
{
    // Get the current node id we're processing.
    int node_id = get_global_id(0);
    // Calculate the first delta value in the network.
    if (targets != 0) {
        layerDeltas[node_id] = 
            (layerOutputs[node_id] - targets[node_id]) 
            * activationDerivative(layerSums[node_id]);
    }
    // Calculate recursively from the next layer.
    else {
        float sum = 0.0;
        for (int k=0; k<nextLayerNodeCount; k++) 
        {
            sum += nextLayerWeights[node_id * nextLayerNodeCount + k] * nextLayerDeltas[k];
        }
        layerDeltas[node_id] = sum * activationDerivative(layerSums[node_id]);
    }
    // Update weights according to the learning rate.
    for (int i=0; i<previousLayerNodeCount; i++) {
        int weightIndex = i * layerNodeCount + node_id;
        layerWeights[weightIndex] -= 
            // learning rate
            0.1 *
            previousLayerOutputs[i] * layerDeltas[node_id];
    }
}

