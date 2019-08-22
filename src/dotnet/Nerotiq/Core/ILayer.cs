using System;
using Nerotiq.Util.Data;
using OpenCL.Net;

namespace Nerotiq.Core
{
    /**
     * Represents the instance of a layer within a network.
     */
    public interface ILayer : IDisposable
    {
        /**
         * The number of nodes in this layer.
         */
        int NodeCount { get; }

        /**
         * The dimensionality of the layer.
         * Represents the lengths with support for muliple dimensions. (e.g. for CNNs)
         */
        ushort[] Dimensionality { get; }

        /**
         * Gets the layer's output buffer (useful for linking layers).
         * This will match the node count in length. 
         */
        GpuMatrix Outputs { get; }
        
        /**
         * Gets the layer's delta buffer (useful for linking layers).
         * This will match the node count in length. 
         */
        GpuMatrix Deltas { get; }
        
        /**
         * Gets the layer's weight buffer (useful for linking layers).
         * This will match the node count in length. 
         */
        GpuMatrix Weights { get; }

        /**
         * Reference to the previous layer set when building the network.
         */
        ILayer Previous { set; }

        /**
         * Reference to the next layer set when building the network.
         */
        ILayer Next { set; }

        /**
         * 
         */
        void ForwardPass(ExecutionSequence executionSequence);

        /**
         * 
         */
        void BackwardPass(ExecutionSequence executionSequence);

        /**
         * 
         */
        double[] GetOutputs(ExecutionSequence executionSequence);

    }
}