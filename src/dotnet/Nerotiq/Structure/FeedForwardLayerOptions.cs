using Nerotiq.Math.Activation;

namespace Nerotiq.Structure {
    public class FeedForwardLayerOptions : ILayerConfig {
        public ushort[] Dimensionality { get; set; }
        public ushort[] FromDimensionality { get; set; }
        public IActivationOptions ActivationOptions { get; set; }

        public ILayer CreateLayer(ExecutionContext executionContext, bool finalLayer) {
            return new FeedForwardLayer(executionContext, this, finalLayer);
        }
    }
}