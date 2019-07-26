using System.IO;
using Nerotiq.Util;

namespace Nerotiq.Math.Activation {
    public class ReluActivation : IActivation {

        private static readonly string _source;
        static ReluActivation()
        {
            _source = SourceLoader.Read("Nerotiq.activation.relu.clh");
        }

        public string Source => _source;
    }
}