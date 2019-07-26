using System;
using OpenCL.Net;

namespace Nerotiq.Structure
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
         * 
         */
        void ForwardPass(ExecutionSequence executionSequence);

        /**
         * 
         */
        void BackwardPass(ExecutionSequence executionSequence);
    }
}