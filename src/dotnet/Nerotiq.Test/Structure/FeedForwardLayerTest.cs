using System;
using Xunit;
using Nerotiq.Math.Activation;
using Nerotiq.Structure;

namespace Nerotiq.Test.Basic
{
    public class FeedForwardLayerTest : IClassFixture<TestScaffold>
    {
        private readonly TestScaffold _scaffold;

        public FeedForwardLayerTest(TestScaffold scaffold) {
            _scaffold = scaffold;
        }

        private static readonly FeedForwardLayerOptions _ffOpt1 = new FeedForwardLayerOptions 
        {
            Dimensionality = new ushort[] { 10 },
            ActivationOptions = new ReluActivationOptions()
        };

        [Fact]
        public void FeedForwardTest()
        {
            var input = _scaffold.CreateInput(10);
            var layer = _ffOpt1.CreateLayer(_scaffold.Context);
        }
    }
}
