/*
 * Standard feed-forward neural network layer weight update.
 */


/**
 * Updates the weights and biases for a layer.
 *
 * previousLayerNodeCount: number of nodes in the previous layer.
 * layerNodeCount: number of nodes in the current layer.
 * layerDeltas: pointer to layer delta value vector.
 * layerWeights: pointer to layer weight vector.
 * layerBiases: pointer to layer bias vector.
 * previousLayerOutputs: pointer to the previous layer's output vector. 
 */
__kernel void update(
    double learningRate,// 0
    uint previousLayerNodeCount, // 1
    uint layerNodeCount, // 2
    __global float *layerDeltas, // 3
    __global float *layerWeights, // 4
    __global float *layerBiases, // 5
    __global float *previousLayerOutputs // 6
)
{
    // Get the current node id we're processing.
    int node_id = get_global_id(0);
    // Update weights according to the learning rate.
    for (int i=0; i<previousLayerNodeCount; i++) {
        int weightIndex = i * layerNodeCount + node_id;
        layerWeights[weightIndex] -= 
            // learning rate
            0.1 *
            previousLayerOutputs[i] * layerDeltas[node_id];
    }
    layerBiases[node_id] -=
        0.1 *
        layerDeltas[node_id];
}

