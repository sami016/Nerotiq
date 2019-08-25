using System;
using Xunit;
using Xunit.Abstractions;
using Nerotiq.Math.Activation;
using Nerotiq.Core;
using Nerotiq.Core.FeedForward;
using Nerotiq.Core.Input;
using FluentAssertions;

namespace Nerotiq.Test.Core
{
    public class FeedForwardLayerTest : IClassFixture<TestScaffold>
    {
        private readonly TestScaffold _scaffold;
        private readonly ITestOutputHelper _output;

        private double tolerance = 0.001;

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
        
        private static readonly FeedForwardLayerOptions _ffOpt2_A = new FeedForwardLayerOptions 
        {
            Dimensionality = new ushort[] { 1 },
            FromDimensionality = new ushort[] { 1 },
            ActivationOptions = new ReluActivationOptions()
        };
        
        
        private static readonly FeedForwardLayerOptions _ffOpt2_B = new FeedForwardLayerOptions 
        {
            Dimensionality = new ushort[] { 1 },
            FromDimensionality = new ushort[] { 1 },
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
            input.SetInputs(ExecutionSequence, new double[] {
                2.0,
                1.0
            });
            _output.WriteLine("Setting weights");
            layer.SetWeights(ExecutionSequence, new double[] {
                1.0, 1.0, 1.0,
                1.0, -1.0, -2.0
            });
            layer.SetBiases(ExecutionSequence, new double[] {
                4.0,
                -2.0,
                1.0,
            });

            _output.WriteLine("Enqueing forward pass");
            layer.ForwardPass(ExecutionSequence);
            ExecutionSequence.FinishExecution();

            var inputs = input.GetOutputs(ExecutionSequence);
            var outputs = layer.GetOutputs(ExecutionSequence);

            outputs[0].Should().Be(7.0);
            outputs[1].Should().Be(0.0);
            outputs[2].Should().Be(1.0);
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
            input.SetInputs(ExecutionSequence, new double[] {
                2.0,
                1.0
            });
            _output.WriteLine("Setting weights");
            layer.SetWeights(ExecutionSequence, new double[] {
                // Weights from node 0 in the previous layer.
                1.0, 1.0, 1.0,
                // Weights from node 1 in the previous layer.
                1.0, -1.0, -2.0
            });
            layer.SetBiases(ExecutionSequence, new double[] {
                4.0,
                -2.0,
                1.0,
            });


            layer.SetTargets(ExecutionSequence, new double[] {
                // Target value for node 0 in the output layer.
                4.0,
                // Target value for node 1 in the output layer.
                10.0,
                // Target value for node 2 in the output layer.
                -2.0
            });

            _output.WriteLine("Enqueing forward pass");
            layer.ForwardPass(ExecutionSequence);

            _output.WriteLine("Enqueing backward pass");
            layer.BackwardPass(ExecutionSequence);
            layer.UpdateParameters(ExecutionSequence);
            ExecutionSequence.FinishExecution();

            var inputs = input.GetOutputs(ExecutionSequence);
            var outputs = layer.GetOutputs(ExecutionSequence);
            var weights = layer.GetWeights(ExecutionSequence);
            var biases = layer.GetBiases(ExecutionSequence);
            var deltas = layer.GetDeltas(ExecutionSequence);

            outputs[0].Should().BeApproximately(7.0, tolerance);
            outputs[1].Should().BeApproximately(0.0, tolerance);
            outputs[2].Should().BeApproximately(1.0, tolerance);

            deltas[0].Should().BeApproximately(3.0, tolerance);
            deltas[1].Should().BeApproximately(0.0, tolerance);
            deltas[2].Should().BeApproximately(3.0, tolerance);

            weights[0].Should().BeApproximately(0.4, tolerance);
            weights[1].Should().BeApproximately(1.0, tolerance);
            weights[2].Should().BeApproximately(0.4, tolerance);

            weights[3].Should().BeApproximately(0.7, tolerance);
            weights[4].Should().BeApproximately(-1.0, tolerance);
            weights[5].Should().BeApproximately(-2.3, tolerance);
        }

        
        /// <summary>
        /// A single rectified layer to test backward propagation.
        /// Same set up as Execution_Working.
        /// </summary>
        [Fact]
        public void Backpropagation_Deep_Working()
        {
            _output.WriteLine("Starting...");
            var input = _scaffold.CreateInput(1);
            _output.WriteLine("Created input");
            var layer1 = _ffOpt2_A.CreateLayer(_scaffold.Context, false) as FeedForwardLayer;
            var layer2 = _ffOpt2_B.CreateLayer(_scaffold.Context, true) as FeedForwardLayer;
            _output.WriteLine("Created layer");
            // Link the layers together.
            layer1.Previous = input;
            layer1.Next = layer2;
            layer2.Previous = layer1;
            _output.WriteLine("Linked input to layer");

            _output.WriteLine("Setting inputs");
            input.SetInputs(ExecutionSequence, new double[] {
                1.0
            });
            _output.WriteLine("Setting weights");
            layer1.SetWeights(ExecutionSequence, new double[] {
                1.0
            });
            layer1.SetBiases(ExecutionSequence, new double[] {
                1.0
            });
            
            layer2.SetWeights(ExecutionSequence, new double[] {
                2.0
            });
            layer2.SetBiases(ExecutionSequence, new double[] {
                2.0
            });


            layer2.SetTargets(ExecutionSequence, new double[] {
                10.0
            });

            _output.WriteLine("Enqueing forward pass");
            layer1.ForwardPass(ExecutionSequence);
            layer2.ForwardPass(ExecutionSequence);

            _output.WriteLine("Enqueing backward pass");
            layer2.BackwardPass(ExecutionSequence);
            layer1.BackwardPass(ExecutionSequence);
            
            layer2.UpdateParameters(ExecutionSequence);
            layer1.UpdateParameters(ExecutionSequence);
            ExecutionSequence.FinishExecution();

            var inputs = input.GetOutputs(ExecutionSequence);
            var outputs1 = layer1.GetOutputs(ExecutionSequence);
            var weights1 = layer1.GetWeights(ExecutionSequence);
            var biases1 = layer1.GetBiases(ExecutionSequence);
            var deltas1 = layer1.GetDeltas(ExecutionSequence);
            var outputs2 = layer2.GetOutputs(ExecutionSequence);
            var weights2 = layer2.GetWeights(ExecutionSequence);
            var biases2 = layer2.GetBiases(ExecutionSequence);
            var deltas2 = layer2.GetDeltas(ExecutionSequence);

            outputs1[0].Should().BeApproximately(2, tolerance);
            outputs2[0].Should().BeApproximately(6, tolerance);

            deltas2[0].Should().BeApproximately(-4f, tolerance);
            deltas1[0].Should().BeApproximately(-8f, tolerance);
            // deltas[0].Should().Be(3);

            // weights[0].Should().Be(0.4f);
            
        }
    }
}
