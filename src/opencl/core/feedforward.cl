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
    uint previousLayerNodeCount,
    uint layerNodeCount,
    __global float *previousLayerOutputs,
    __global float *layerWeights,
    __global float *layerBiases,
    __global float *layerSums,
    __global float *layerOutputs
)
{
    // Get the current node id we're processing.
    int node_id = get_global_id(0);
    // Sum up the product of the inputs and the weights.
    float sum = 0.0;
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