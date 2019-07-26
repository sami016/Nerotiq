using Nerotiq;
using Nerotiq.Structure;
using Xunit;

namespace Nerotiq.Test
{
    public class TestScaffold {

        public ExecutionContext Context { get; }
        public ExecutionSequence ExecutionSequence { get; }

        public TestScaffold() {
            Context = new ExecutionContext();
            ExecutionSequence = new ExecutionSequence(Context);
        }

        public InputLayer CreateInput(int width = 10) {
            return new InputLayerOptions {
                Dimensionality = new ushort[] { width }
            }.CreateLayer(Context) as InputLayer;
        }

    }
}