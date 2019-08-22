using System;
using Nerotiq.Util.Data;
using OpenCL.Net;

namespace Nerotiq.Core.Input {
    public class InputLayer : ILayer, IInput {
        
        public int NodeCount { get; }
        public ushort[] Dimensionality { get; }
        public ILayer Previous { get; set; }
        public ILayer Next { get; set; }

        public GpuMatrix Outputs => _inputs;
        public GpuMatrix Deltas => null;
        public GpuMatrix Weights => null;

        private readonly ExecutionContext _executionContext;
        private readonly GpuMatrix _inputs;

        public InputLayer(ExecutionContext executionContext, InputLayerOptions options) 
        {
            _executionContext = executionContext;
            Dimensionality = options.Dimensionality;
            NodeCount = 1;
            foreach (var i in options.Dimensionality) {
                NodeCount *= i;
            }
            _inputs = new GpuMatrix((ushort)NodeCount, 1, executionContext);
        }

        public void SetInputs(ExecutionSequence executionSequence, double[] inputs)
        {
            if (inputs.Length != NodeCount) {
                throw new ArgumentException($"input array length ({inputs.Length}) does not match input layer size ({NodeCount})", nameof(inputs));
            }
            _inputs.Update(inputs, executionSequence);
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

        public double[] GetOutputs(ExecutionSequence executionSequence)
        {
            using (_inputs.Read(executionSequence))
            {
                return _inputs.InMemoryData;
            }
        }
    }
}