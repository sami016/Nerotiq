
namespace Nerotiq.Core 
{
    public interface ILayerConfig 
    {
        ILayer CreateLayer(ExecutionContext executionContext, bool finalLayer);
    }
}