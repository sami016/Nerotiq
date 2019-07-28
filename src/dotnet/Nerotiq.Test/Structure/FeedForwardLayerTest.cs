using System;
using Xunit;
using Xunit.Abstractions;
using Nerotiq.Math.Activation;
using Nerotiq.Structure;
using FluentAssertions;

namespace Nerotiq.Test.Basic
{
    public class FeedForwardLayerTest : IClassFixture<TestScaffold>
    {
        private readonly TestScaffold _scaffold;
        private readonly ITestOutputHelper _output;

        private ExecutionSequence ExecutionSequence => _scaffold.ExecutionSequence;

        public FeedForwardLayerTest(TestScaffold scaffold, ITestOutputHelper output) {
            _scaffold = scaffold;
            _output = output;
        }

        private static readonly FeedForwardLayerOptions _ffOpt1 = new FeedForwardLayerOptions 
        {
            Dimensionality = new ushort[] { 3 },
            FromDimensionality = new ushort[] { 2 },
            ActivationOptions = new ReluActivationOptions()
        };

        [Fact]
        public void FeedForwardTest()
        {
            _output.WriteLine("Starting...");
            var input = _scaffold.CreateInput(2);
            _output.WriteLine("Created input");
            var layer = _ffOpt1.CreateLayer(_scaffold.Context) as FeedForwardLayer;
            _output.WriteLine("Created layer");
            // Link the layers together.
            layer.Previous = input;
            _output.WriteLine("Linked input to layer");

            _output.WriteLine("Setting inputs");
            input.SetInputs(ExecutionSequence, new float[] {
                2,
                1f
            });
            _output.WriteLine("Setting weights");
            layer.SetWeights(ExecutionSequence, new float[] {
                1, 1, 1,
                1, -1, -2
            });

            _output.WriteLine("Executing forward pass");
            layer.ForwardPass(ExecutionSequence);
            ExecutionSequence.FinishExecution();

            // layer.SetOutputs(ExecutionSequence, new float[] {
            //     3, 1, 0
            // });

            var inputs = input.GetOutputs(ExecutionSequence);
            var outputs = layer.GetOutputs(ExecutionSequence);

            {            
                var i = 0;
                _output.WriteLine("Inputs:");
                foreach (var o in inputs) {
                    _output.WriteLine($"{i}: {o}");
                    i++;
                }
            }
            {            
                var i = 0;
                _output.WriteLine("Layer outputs:");
                foreach (var o in outputs) {
                    _output.WriteLine($"{i}: {o}");
                    i++;
                }
            }

            outputs[0].Should().Be(3);
            outputs[1].Should().Be(1);
            outputs[2].Should().Be(0);
        }
    }
}
