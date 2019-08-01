using Nerotiq.Math.Activation;

namespace Nerotiq.Core.FeedForward {
    public class FeedForwardLayerOptions : ILayerConfig {

        /// <summary>
        /// The dimensionality of the layer.
        /// </summary>
        /// <value></value>
        public ushort[] Dimensionality { get; set; }

        /// <summary>
        /// The dimensionality of the previous layer.
        /// </summary>
        /// <value></value>
        public ushort[] FromDimensionality { get; set; }

        /// <summary>
        /// Options for the activation function to be used.
        /// </summary>
        /// <value>activation options</value>
        public IActivationOptions ActivationOptions { get; set; }

        /// <summary>
        /// Options for how the weights and biases are to be updated.
        /// </summary>
        /// <value>update options</value>
        public IOption<IFeedForwardUpdate> UpdateOptions { get; set; }

        public ILayer CreateLayer(ExecutionContext executionContext, bool finalLayer) {
            return new FeedForwardLayer(executionContext, this, finalLayer);
        }
    }
}