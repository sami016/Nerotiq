namespace Nerotiq.Structure 
{
    /**
     * Interface for a layer that can act as an output.
     * Supports querying the output values.
     * Supports setting of target values.
     */
    public interface IOutput {
        /**
         * Gets the current outputs of the layer.
         */
        float[] GetOutputs(ExecutionSequence executionSequence);

        /**
         * Sets the targets for the outputs layer.
         * This should only be called if the layer is the final layer in the network.
         */
        void SetTargets(ExecutionSequence executionSequence, float[] targets);
    }
}