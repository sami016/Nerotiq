using Nerotiq;
using Nerotiq.Core;
using Nerotiq.Core.FeedForward;
using Nerotiq.Core.Input;
using Xunit;

[assembly: CollectionBehavior(DisableTestParallelization = true)]

namespace Nerotiq.Test
{
    public class TestScaffold {

        public ExecutionContext Context { get; }
        public ExecutionSequence ExecutionSequence { get; }

        public TestScaffold() {
            Context = new ExecutionContext();
            ExecutionSequence = new ExecutionSequence(Context);
        }

        public InputLayer CreateInput(ushort width = 10) {
            return new InputLayerOptions {
                Dimensionality = new ushort[] { width }
            }.CreateLayer(Context, false) as InputLayer;
        }

    }
}