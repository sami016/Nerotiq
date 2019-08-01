using System;
using OpenCL.Net;

namespace Nerotiq.Core.Input {
    public class InputLayer : ILayer, IInput {
        
        public int NodeCount { get; }
        public ushort[] Dimensionality { get; }
        public ILayer Previous { get; set; }
        public ILayer Next { get; set; }

        public IMem<float> Outputs => _inputs;
        public IMem<float> Deltas => null;
        public IMem<float> Weights => null;

        private readonly ExecutionContext _executionContext;
        private readonly IMem<float> _inputs;

        public InputLayer(ExecutionContext executionContext, InputLayerOptions options) 
        {
            _executionContext = executionContext;
            Dimensionality = options.Dimensionality;
            NodeCount = 1;
            foreach (var i in options.Dimensionality) {
                NodeCount *= i;
            }
            _inputs = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                NodeCount, 
                out var error
            );
        }

        public void SetInputs(ExecutionSequence executionSequence, float[] inputs)
        {
            if (inputs.Length != NodeCount) {
                throw new ArgumentException($"input array length ({inputs.Length}) does not match input layer size ({NodeCount})", nameof(inputs));
            }
            executionSequence.EnqueueWriteBuffer(
                _inputs,
                0,
                inputs.Length,
                inputs
            );
        }

        public void ForwardPass(ExecutionSequence executionSequence)
        {
            // Nothing to be done.
        }

        public void BackwardPass(ExecutionSequence executionSequence)
        {
            // Nothing to be done.
        }

        public void Dispose()
        {
        }

        public float[] GetOutputs(ExecutionSequence executionSequence)
        {
            return executionSequence.ReadBuffer(
                _inputs,
                0,
                NodeCount
            );
        }
    }
}