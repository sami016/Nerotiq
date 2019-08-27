using Nerotiq.Exceptions;

namespace Nerotiq.Core.Input {
    public class InputLayerOptions : ILayerConfig
    {
        public ushort[] Dimensionality { get; set; }

        public ILayer CreateLayer(ExecutionContext executionContext, ushort[] previousLayerDimensionality) {
            // Something has gone wrong if an input layer is the final layer.
            if (previousLayerDimensionality != null) {
                throw new NerotiqException("Input layer must not be the final layer");
            }
            return new InputLayer(executionContext, this);
        }
    }
}