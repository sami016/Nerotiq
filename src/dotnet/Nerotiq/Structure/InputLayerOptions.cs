namespace Nerotiq.Structure {
    public class InputLayerOptions : ILayerConfig
    {
        public ushort[] Dimensionality { get; set; }

        public ILayer CreateLayer(ExecutionContext executionContext) {
            return new InputLayer(executionContext, this);
        }
    }
}