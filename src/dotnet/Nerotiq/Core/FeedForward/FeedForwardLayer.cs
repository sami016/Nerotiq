using System;
using System.Runtime.InteropServices;
using Nerotiq.Exceptions;
using Nerotiq.Math.Activation;
using Nerotiq.Util;
using Nerotiq.Util.Data;
using OpenCL.Net;

namespace Nerotiq.Core.FeedForward {
    public class FeedForwardLayer : ILayer, IOutput
    {
        private static readonly string _source;
        private GpuMatrix _layerSums;
        private GpuMatrix _layerOutputs;
        private GpuMatrix _layerDeltas;
        // This is only defined for the final output layer.
        private GpuMatrix _layerTargets;
        private GpuMatrix _weights;
        private GpuMatrix _biases;
        private Program _program;
        private Kernel _forwardKernel;
        private Kernel _backwardKernel;
        private readonly IActivation _activation;
        private readonly IFeedForwardUpdate _update;

        static FeedForwardLayer() {
            _source = SourceLoader.Read("Nerotiq.core.feedforward.feedforward.cl");
        }

        public FeedForwardLayer(int nodeCount) 
        {
            this.NodeCount = nodeCount;
               
        }
        public int NodeCount { get; set; }

        private readonly int _fromNodeCount;
        private readonly int _previousLayerNodeCount;
        private readonly FeedForwardLayerOptions _options;
        private readonly bool _finalLayer;

        public ushort[] Dimensionality { get; set; }

        private readonly int _weightLength;

        public GpuMatrix Outputs => _layerOutputs;
        public GpuMatrix Deltas => _layerDeltas;
        public GpuMatrix Weights => _weights;
        public GpuMatrix Biases => _biases;

        public FeedForwardLayer(ExecutionContext executionContext, FeedForwardLayerOptions options, bool finalLayer)
        {
            _options = options;
            _finalLayer = finalLayer;
            Dimensionality = options.Dimensionality;
            _weightLength = MatrixHelpers.GetWeightCardinality(options.FromDimensionality, options.Dimensionality);
            NodeCount = MatrixHelpers.GetCardinality(options.Dimensionality);
            _fromNodeCount = MatrixHelpers.GetCardinality(options.FromDimensionality);
            _previousLayerNodeCount = MatrixHelpers.GetCardinality(options.FromDimensionality);
            _activation = (options.ActivationOptions ?? new ReluActivationOptions())
                .Create();
            _update = (options.UpdateOptions ?? new FeedForwardUpdateOptions())
                .Create(executionContext);

            CompileKernels(executionContext);
            AllocateBuffers(executionContext, options);
            SetForwardPassArgs();
            SetBackwardPassArgs();
        }

        private void CompileKernels(ExecutionContext executionContext) 
        {
            var sourceCollection =  SourceLoader.CreateProgramCollection(_activation.Source, _source);
            _program = Cl.CreateProgramWithSource(
                executionContext.OpenClContext, 
                (uint)sourceCollection.Length,
                sourceCollection,
                null,
                out var error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error creating program with source: {error}");
            }
            error = Cl.BuildProgram(_program, 1, new[] { executionContext.Device }, string.Empty, null, IntPtr.Zero);
            if (error != ErrorCode.Success) 
            {
                if (error == ErrorCode.BuildProgramFailure) 
                {
                    var buildInfoLog = Cl.GetProgramBuildInfo(_program, executionContext.Device, ProgramBuildInfo.Log, out var buildInfoError);
                    throw new NerotiqException($"Error building program: {error}: {buildInfoLog}");
                }
                throw new NerotiqException($"Error building program: {error}");
            }

            // Get the kernels.
            _forwardKernel = Cl.CreateKernel(_program, "forwardPass", out error);
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error creating kernel forwardPass: {error}");
            }
            _backwardKernel = Cl.CreateKernel(_program, "backwardPass", out error);
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error creating kernel backwardPass: {error}");
            }
        }

        private void AllocateBuffers(ExecutionContext executionContext, FeedForwardLayerOptions options) 
        {
            try {
                _layerSums = new GpuMatrix((ushort)NodeCount, 1, executionContext);
            } catch (Exception ex)
            {
                throw new NerotiqException($"Error allocating sum buffer", ex);
            }
            try {
                _layerOutputs = new GpuMatrix((ushort)NodeCount, 1, executionContext);
            } catch (Exception ex)
            {
                throw new NerotiqException($"Error allocating output buffer", ex);
            }
            try {
                _layerDeltas = new GpuMatrix((ushort)NodeCount, 1, executionContext);
            } catch (Exception ex)
            {
                throw new NerotiqException($"Error allocating delta buffer", ex);
            }

            // Parameters
            
            try {
                _weights = new GpuMatrix((ushort)NodeCount, (ushort)_fromNodeCount, executionContext);
            } catch (Exception ex)
            {
                throw new NerotiqException($"Error allocating weight buffer", ex);
            }
            
            try {
                _biases = new GpuMatrix((ushort)NodeCount, 1, executionContext);
            } catch (Exception ex)
            {
                throw new NerotiqException($"Error allocating bias buffer", ex);
            }
        }

        private void SetForwardPassArgs() 
        {
            // Arg 0: previousLayerNodeCount (uint)
            ClHelpers.SetKernelArg(
                _forwardKernel,
                0,
                (uint)_previousLayerNodeCount
            );
            // Arg 1: layerNodeCount (uint)
            ClHelpers.SetKernelArg(
                _forwardKernel,
                1,
                (uint)NodeCount
            );
            // Arg 3: layerWeights (float*)
            _weights.SetKernelArg(_forwardKernel, 3);
            // Arg 4: layerBiases (float*)
            _biases.SetKernelArg(_forwardKernel, 4);
            // Arg 5: layerSums (float*)
            _layerSums.SetKernelArg(_forwardKernel, 5);
            // Arg 6: layerOutputs (float*)
            _layerOutputs.SetKernelArg(_forwardKernel, 6);
        }

           
        private void SetBackwardPassArgs() 
        {
            // Arg 1: layerNodeCount (uint)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                1,
                (uint)NodeCount
            );
             // Arg 2: nextLayerNodeCount (uint)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                2,
                (uint)0
            );
            // Arg 3: layerSums (float*)
            _layerSums.SetKernelArg(_backwardKernel, 3);
            // Arg 4: layerOutputs (float*)
            _layerOutputs.SetKernelArg(_backwardKernel, 4);
            // Arg 5: layerOutputs (float*)
            _layerDeltas.SetKernelArg(_backwardKernel, 5);
            // Arg 6: layerWeights (float*)
            _weights.SetKernelArg(_backwardKernel, 6);
            // Arg 7: layerBiases (float*)
            _biases.SetKernelArg(_backwardKernel, 7);
            // Arg 8: previousLayerOutputs (float*)
            GpuMatrix.SetNullKernelArg(_backwardKernel, 8);
            // Arg 9: nextLayerDeltas (float*)
            GpuMatrix.SetNullKernelArg(_backwardKernel, 9);
            // // Arg 10: nextLayerWeights (float*)
            GpuMatrix.SetNullKernelArg(_backwardKernel, 10);
            // Arg 11: targets (float*)
            GpuMatrix.SetNullKernelArg(_backwardKernel, 11);
        }

        public ILayer Previous { 
            set {
                // Links to previous layer.

                // Arg 2: previousLayerOutputs (float*)
                value.Outputs.SetKernelArg(
                    _forwardKernel,
                    2
                );

                // Arg 0: previousLayerNodeCount (uint)
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    0,
                    (uint)value.NodeCount
                );
                // Arg 8: previousLayerOutputs (float*)
                value.Outputs.SetKernelArg(
                    _backwardKernel,
                    8
                );

                // Set-up the update scheme, which requires knowledge of the previous layer's outputs.
                _update.SetUp(this, value);
            }
        }

        public ILayer Next {
            set { 
                // Links to the next layer.

                // Arg 2: nextLayerNodeCount (uint)
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    2,
                    (uint)value.NodeCount
                );
                // Arg 9: nextLayerDeltas (float*)
                value.Deltas.SetKernelArg(
                    _backwardKernel,
                    9
                );
                // Arg 10: nextLayerWeights (float*)
                value.Weights.SetKernelArg(
                    _backwardKernel,
                    10
                );
            }
        }

        public void ForwardPass(ExecutionSequence executionSequence)
        {
            executionSequence.EnqueueNDRangeKernel(
                _forwardKernel,
                1,
                null,
                new IntPtr [] { new IntPtr(NodeCount) },
                null
            );
        }

        public void BackwardPass(ExecutionSequence executionSequence)
        {
            executionSequence.EnqueueNDRangeKernel(
                _backwardKernel,
                1,
                null,
                new IntPtr [] { new IntPtr(NodeCount) },
                null
            );
        }

        public void UpdateParameters(ExecutionSequence executionSequence)
        {
            _update.Update(executionSequence);
        }

        public void SetWeights(ExecutionSequence executionSequence, double[] weights)
        {
            if (weights.Length != _weightLength) {
                throw new ArgumentException($"weight array length ({weights.Length}) does not match required ({_weightLength})", nameof(weights));
            }
            _weights.Update(weights, executionSequence);
        }

        public void SetBiases(ExecutionSequence executionSequence, double[] biases)
        {
            _biases.Update(biases, executionSequence);
        }

        public double[] GetOutputs(ExecutionSequence executionSequence)
        {
            using (_layerOutputs.Read(executionSequence)) 
            {
                return _layerOutputs.InMemoryData;
            }
        }
        
        public double[] GetWeights(ExecutionSequence executionSequence)
        {
            using (_weights.Read(executionSequence)) 
            {
                return _weights.InMemoryData;
            }
        }
        
        public double[] GetBiases(ExecutionSequence executionSequence)
        {
            using (_biases.Read(executionSequence)) 
            {
                return _biases.InMemoryData;
            }
        }
        
        public double[] GetDeltas(ExecutionSequence executionSequence)
        {
            using (_layerDeltas.Read(executionSequence)) 
            {
                return _layerDeltas.InMemoryData;
            }
        }
        
        public void SetOutputs(ExecutionSequence executionSequence, double[] outputs)
        {
            _layerOutputs.Update(outputs, executionSequence);
        }
        
        public void SetTargets(ExecutionSequence executionSequence, double[] targets)
        {
            if (targets.Length != NodeCount) {
                throw new NerotiqException($"target value array length ({targets.Length}) does not match layer size ({NodeCount})");
            }
            if (_layerTargets == null) 
            {
                try {
                    _layerTargets = new GpuMatrix((ushort)NodeCount, 1, executionSequence.Context);
                } catch (Exception ex)
                {
                    throw new NerotiqException($"Error allocating delta buffer", ex);
                }
                _layerTargets.SetKernelArg(
                    _backwardKernel,
                    11
                );
            }
            _layerTargets.Update(targets, executionSequence);
        }

        public void Dispose()
        {
            _program.Dispose();
            _forwardKernel.Dispose();
            _backwardKernel.Dispose();
            _layerSums?.Dispose();
            _layerOutputs?.Dispose();
            _weights?.Dispose();
            _biases?.Dispose();
        }

    }
}