using System;
using System.Runtime.InteropServices;
using Nerotiq.Exceptions;
using Nerotiq.Math.Activation;
using Nerotiq.Util;
using OpenCL.Net;

namespace Nerotiq.Core.FeedForward {
    public class FeedForwardLayer : ILayer, IOutput
    {
        private static readonly string _source;
        private IMem<float> _layerSums;
        private IMem<float> _layerOutputs;
        private IMem<float> _layerDeltas;
        // This is only defined for the final output layer.
        private IMem<float> _layerTargets;
        private IMem<float> _weights;
        private IMem<float> _biases;
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

        private readonly int _previousLayerNodeCount;
        private readonly bool _finalLayer;

        public ushort[] Dimensionality { get; set; }

        private readonly int _weightLength;

        public IMem<float> Outputs => _layerOutputs;
        public IMem<float> Deltas => _layerDeltas;
        public IMem<float> Weights => _weights;
        public IMem<float> Biases => _biases;

        public FeedForwardLayer(ExecutionContext executionContext, FeedForwardLayerOptions options, bool finalLayer)
        {
            _finalLayer = finalLayer;
            Dimensionality = options.Dimensionality;
            _weightLength = MatrixHelpers.GetWeightCardinality(options.FromDimensionality, options.Dimensionality);
            NodeCount = MatrixHelpers.GetCardinality(options.Dimensionality);
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
            _program = Cl.CreateProgramWithSource(
                executionContext.OpenClContext, 
                2, 
                SourceLoader.CreateProgramCollection(_activation.Source, _source),
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
            _layerSums = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                NodeCount, 
                out var error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error allocating memory buffer {error}");
            }
            _layerOutputs = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                NodeCount, 
                out error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error allocating memory buffer {error}");
            }
            _layerDeltas = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                NodeCount, 
                out error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error allocating delta buffer {error}");
            }
            if (_finalLayer) {
                _layerTargets = Cl.CreateBuffer<float>(
                    executionContext.OpenClContext, 
                    MemFlags.ReadWrite,
                    NodeCount, 
                    out error
                );
                if (error != ErrorCode.Success) 
                {
                    throw new NerotiqException($"Error allocating targets buffer {error}");
                }
            }

            // Parameters
            _weights = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                _weightLength, 
                out error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error allocating weight buffer {error}");
            }
            _biases = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                NodeCount, 
                out error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error allocating bias buffer {error}");
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
            ClHelpers.SetKernelArg(
                _forwardKernel,
                3,
                new IntPtr(MiscHelpers.IntPtrSize),
                _weights
            );
            // Arg 4: layerBiases (float*)
            ClHelpers.SetKernelArg(
                _forwardKernel,
                4,
                new IntPtr(MiscHelpers.IntPtrSize),
                _biases
            );
            // Arg 5: layerSums (float*)
            ClHelpers.SetKernelArg(
                _forwardKernel,
                5,
                new IntPtr(MiscHelpers.IntPtrSize),
                _layerSums
            );
            // Arg 6: layerOutputs (float*)
            ClHelpers.SetKernelArg(
                _forwardKernel,
                6,
                new IntPtr(MiscHelpers.IntPtrSize),
                _layerOutputs
            );
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
            ClHelpers.SetKernelArg(
                _backwardKernel,
                3,
                new IntPtr(MiscHelpers.IntPtrSize),
                _layerSums
            );
            // Arg 4: layerOutputs (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                4,
                new IntPtr(MiscHelpers.IntPtrSize),
                _layerOutputs
            );
            // Arg 5: layerOutputs (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                5,
                new IntPtr(MiscHelpers.IntPtrSize),
                _layerDeltas
            );
            // Arg 6: layerWeights (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                6,
                new IntPtr(MiscHelpers.IntPtrSize),
                _weights
            );
            // Arg 7: layerBiases (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                7,
                new IntPtr(MiscHelpers.IntPtrSize),
                _biases
            );
            // Arg 8: previousLayerOutputs (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                8,
                new IntPtr(MiscHelpers.IntPtrSize),
                new IntPtr(0)
            );
            // Arg 9: nextLayerDeltas (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                9,
                new IntPtr(MiscHelpers.IntPtrSize),
                new IntPtr(0)
            );
            // Arg 10: nextLayerWeights (float*)
            ClHelpers.SetKernelArg(
                _backwardKernel,
                10,
                new IntPtr(MiscHelpers.IntPtrSize),
                new IntPtr(0)
            );
            // Arg 11: targets (float*)
            if (_finalLayer) {
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    11,
                    new IntPtr(MiscHelpers.IntPtrSize),
                    _layerTargets
                );
            } else {
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    11,
                    new IntPtr(MiscHelpers.IntPtrSize),
                    new IntPtr(0)
                );
            }
        }

        public ILayer Previous { 
            set {
                // Links to previous layer.

                // Arg 2: previousLayerOutputs (float*)
                ClHelpers.SetKernelArg(
                    _forwardKernel,
                    2,
                    new IntPtr(MiscHelpers.IntPtrSize),
                    value.Outputs
                );

                
                // Arg 0: previousLayerNodeCount (uint)
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    0,
                    (uint)value.NodeCount
                );
                // Arg 8: previousLayerOutputs (float*)
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    8,
                    new IntPtr(MiscHelpers.IntPtrSize),
                    value.Outputs
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
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    9,
                    new IntPtr(MiscHelpers.IntPtrSize),
                    value.Deltas
                );
                // Arg 10: nextLayerWeights (float*)
                ClHelpers.SetKernelArg(
                    _backwardKernel,
                    10,
                    new IntPtr(MiscHelpers.IntPtrSize),
                    value.Weights
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

        public void SetWeights(ExecutionSequence executionSequence, float[] weights)
        {
            if (weights.Length != _weightLength) {
                throw new ArgumentException($"weight array length ({weights.Length}) does not match required ({_weightLength})", nameof(weights));
            }
            executionSequence.EnqueueWriteBuffer(
                _weights,
                0,
                weights.Length,
                weights
            );
        }

        public void SetBiases(ExecutionSequence executionSequence, float[] biases)
        {
            executionSequence.EnqueueWriteBuffer(
                _biases,
                0,
                biases.Length,
                biases
            );
        }

        public float[] GetOutputs(ExecutionSequence executionSequence)
        {
            return executionSequence.ReadBuffer(
                _layerOutputs,
                0,
                NodeCount
            );
        }
        
        public float[] GetWeights(ExecutionSequence executionSequence)
        {
            return executionSequence.ReadBuffer(
                _weights,
                0,
                _weightLength
            );
        }
        
        public float[] GetBiases(ExecutionSequence executionSequence)
        {
            return executionSequence.ReadBuffer(
                _biases,
                0,
                NodeCount
            );
        }
        
        public float[] GetDeltas(ExecutionSequence executionSequence)
        {
            return executionSequence.ReadBuffer(
                _layerDeltas,
                0,
                NodeCount
            );
        }
        
        public void SetOutputs(ExecutionSequence executionSequence, float[] outputs)
        {
            executionSequence.EnqueueWriteBuffer(
                _layerOutputs,
                0,
                NodeCount,
                outputs
            );
        }
        
        public void SetTargets(ExecutionSequence executionSequence, float[] targets)
        {
            if (targets.Length != NodeCount) {
                throw new NerotiqException($"target value array length ({targets.Length}) does not match layer size ({NodeCount})");
            }
            executionSequence.EnqueueWriteBuffer(
                _layerTargets,
                0,
                NodeCount,
                targets
            );
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