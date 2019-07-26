using OpenCL.Net;

namespace Nerotiq.Structure {
    public class InputLayer : ILayer, IInput {
        
        public int NodeCount { get; }
        public ushort[] Dimensionality { get; }
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
            executionSequence.EnqueueWriteBuffer(
                _inputs,
                Bool.True,
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
    }
}