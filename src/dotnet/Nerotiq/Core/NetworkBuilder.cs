using System.Collections.Generic;
using Nerotiq.Core.Input;
using System.Linq;

namespace Nerotiq.Core
{
    public class NetworkBuilder
    {
        private readonly ExecutionContext _executionContext;
        private readonly ushort[] _dimensionality;
        
        private readonly ILayer _inputLayer;
        private readonly IList<ILayer> _allLayers = new List<ILayer>();
        private ILayer _currentLayer;

        public static NetworkBuilder CreateWithInputs(ExecutionContext executionContext, ushort[] dimensions)
        {
            return new NetworkBuilder(executionContext, dimensions);
        }

        public NetworkBuilder AddLayer(ILayer layer)
        {
            var previousLayer = _currentLayer;
            previousLayer.Next = layer;
            layer.Previous = previousLayer;
            _currentLayer = layer;
            _allLayers.Add(layer);
            return this;
        }
        
        public NetworkBuilder AddLayer(ILayerConfig layerConfig)
        {
            return AddLayer(layerConfig.CreateLayer(_executionContext, _currentLayer.Dimensionality));
        }

        private NetworkBuilder(ExecutionContext executionContext, ushort[] dimensionality) 
        {
            _executionContext = executionContext;
            _dimensionality = dimensionality;
            _inputLayer =  new InputLayer(executionContext, new InputLayerOptions {
                Dimensionality = dimensionality
            });
            _currentLayer = _inputLayer;
            _allLayers.Add(_currentLayer);
        }

        public Network Build()
        {
            return new Network(_executionContext, _allLayers.ToArray());
        }
    }
}