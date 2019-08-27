using System;
using System.Linq;
using Nerotiq.Exceptions;

namespace Nerotiq.Core
{
    /**
     * A network consisting of a sequence of layers.
     */
    public class Network
    {
        private readonly ExecutionContext _executionContext;

        public ILayer[] Layers { get; set; }

        public Network(ExecutionContext executionContext, ILayer[] layers) {
            _executionContext = executionContext;
            Layers = layers;
        }

        public double[] GetInputs(ExecutionSequence executionSequence) {
            var firstLayer = Layers.First();
            return firstLayer.GetOutputs(executionSequence);
        }

        public double[] GetOutputs(ExecutionSequence executionSequence) {
            var lastLayer = Layers.Last();
            return lastLayer.GetOutputs(executionSequence);
        }

        public void SetInputs(ExecutionSequence executionSequence, double[] inputs) {
            if (Layers.Length < 1) {
                throw new NerotiqException("Network does not have an input layer");
            }
            (Layers[0] as IInput)?.SetInputs(executionSequence, inputs);
        }

        public void ExecuteForwardPass(ExecutionSequence executionSequence) {
            // Forward pass.
            for (var i=0; i<Layers.Length; i++) {
                Layers[i].ForwardPass(executionSequence);
            }
        }

        public void ExecuteBackwardPass(ExecutionSequence executionSequence) {
            // Backward pass.
            for (var i=Layers.Length-1; i>=0; i--) {
                Layers[i].BackwardPass(executionSequence);
            }
        }
    }
}