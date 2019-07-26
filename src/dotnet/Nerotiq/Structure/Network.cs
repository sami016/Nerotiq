using System;
using Nerotiq.Exceptions;

namespace Nerotiq.Structure
{
    /**
     * A network consisting of a sequence of layers.
     */
    public class Network
    {
        private readonly ExecutionSequence _executionRunner;

        public ILayer[] Layers { get; set; }

        public Network(ExecutionSequence executionRunner) {
            _executionRunner = executionRunner;
        }

        public float[] GetInputs() {
            throw new NotImplementedException();
        }

        public float[] GetOutputs() {
            throw new NotImplementedException();
        }

        private void SetInputs(ExecutionSequence executionSequence, float[] inputs) {
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