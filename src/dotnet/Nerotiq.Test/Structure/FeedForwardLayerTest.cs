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

        /// <summary>
        /// A single rectified layer to test forward execution.
        /// </summary>
        [Fact]
        public void Execution_Working()
        {
            _output.WriteLine("Starting...");
            var input = _scaffold.CreateInput(2);
            _output.WriteLine("Created input");
            var layer = _ffOpt1.CreateLayer(_scaffold.Context, true) as FeedForwardLayer;
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
            layer.SetBiases(ExecutionSequence, new float[] {
                4,
                -2,
                1,
            });

            _output.WriteLine("Enqueing forward pass");
            layer.ForwardPass(ExecutionSequence);
            ExecutionSequence.FinishExecution();

            var inputs = input.GetOutputs(ExecutionSequence);
            var outputs = layer.GetOutputs(ExecutionSequence);

            outputs[0].Should().Be(7);
            outputs[1].Should().Be(0);
            outputs[2].Should().Be(1);
        }
        
        /// <summary>
        /// A single rectified layer to test backward propagation.
        /// Same set up as Execution_Working.
        /// </summary>
        [Fact]
        public void Backpropagation_Working()
        {
            _output.WriteLine("Starting...");
            var input = _scaffold.CreateInput(2);
            _output.WriteLine("Created input");
            var layer = _ffOpt1.CreateLayer(_scaffold.Context, true) as FeedForwardLayer;
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
                // Weights from node 0 in the previous layer.
                1, 1, 1,
                // Weights from node 1 in the previous layer.
                1, -1, -2
            });
            layer.SetBiases(ExecutionSequence, new float[] {
                4,
                -2,
                1,
            });


            layer.SetTargets(ExecutionSequence, new float[] {
                // Target value for node 0 in the output layer.
                4,
                // Target value for node 1 in the output layer.
                10,
                // Target value for node 2 in the output layer.
                -2
            });

            _output.WriteLine("Enqueing forward pass");
            layer.ForwardPass(ExecutionSequence);

            _output.WriteLine("Enqueing backward pass");
            layer.BackwardPass(ExecutionSequence);
            ExecutionSequence.FinishExecution();

            var inputs = input.GetOutputs(ExecutionSequence);
            var outputs = layer.GetOutputs(ExecutionSequence);
            var weights = layer.GetWeights(ExecutionSequence);
            var biases = layer.GetBiases(ExecutionSequence);
            var deltas = layer.GetDeltas(ExecutionSequence);

            outputs[0].Should().Be(7);
            outputs[1].Should().Be(0);
            outputs[2].Should().Be(1);

            deltas[0].Should().Be(3);
            deltas[1].Should().Be(0);
            deltas[2].Should().Be(3);

            weights[0].Should().Be(0.4f);
            weights[1].Should().Be(1f);
            weights[2].Should().Be(0.4f);

            weights[3].Should().Be(0.7f);
            weights[4].Should().Be(-1f);
            weights[5].Should().Be(-2.3f);
        }
    }
}
