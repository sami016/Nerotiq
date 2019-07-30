namespace Nerotiq.Structure {
    public interface ILayerConfig 
    {
        ILayer CreateLayer(ExecutionContext executionContext, bool finalLayer);
    }
}