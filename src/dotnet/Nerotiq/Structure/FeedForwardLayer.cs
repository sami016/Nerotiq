using System;
using System.Runtime.InteropServices;
using Nerotiq.Exceptions;
using Nerotiq.Math.Activation;
using Nerotiq.Util;
using OpenCL.Net;

namespace Nerotiq.Structure {
    public class FeedForwardLayer : ILayer
    {
        private static readonly string _source;
        private IMem<float> _layerSums;
        private IMem<float> _layerOutputs;
        private IMem<float> _weights;
        private IMem<float> _biases;
        private Program _program;
        private Kernel _forwardKernel;
        private Kernel _backwardKernel;
        private readonly IActivation _activation;

        static FeedForwardLayer() {
            _source = SourceLoader.Read("Nerotiq.core.feedforward.cl");
        }

        public FeedForwardLayer(int nodeCount) 
        {
            this.NodeCount = nodeCount;
               
        }
                public int NodeCount { get; set; }

        private readonly int _previousLayerNodeCount;

        public ushort[] Dimensionality { get; set; }

        private readonly int _weightLength;

        public IMem<float> Outputs => _layerOutputs;

        public FeedForwardLayer(ExecutionContext executionContext, FeedForwardLayerOptions options)
        {
            Dimensionality = options.Dimensionality;
            _weightLength = MatrixHelpers.GetWeightCardinality(options.FromDimensionality, options.Dimensionality);
            NodeCount = MatrixHelpers.GetCardinality(options.Dimensionality);
            _previousLayerNodeCount = MatrixHelpers.GetCardinality(options.FromDimensionality);
            _activation = (options.ActivationOptions ?? new ReluActivationOptions())
                .Create();

            CompileKernels(executionContext);
            AllocateBuffers(executionContext, options);
            SetForwardPassArgs();
        }

        private void CompileKernels(ExecutionContext executionContext) 
        {
            _program = Cl.CreateProgramWithSource(
                executionContext.OpenClContext, 
                2, 
                new[] { _activation.Source, _source },
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
            // _backwardKernel = Cl.CreateKernel(_program, "backwardPass", out error);
            // if (error != ErrorCode.Success) 
            // {
            //     throw new NerotiqException($"Error creating kernel backwardPass: {error}");
            // }
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
        
        public void SetOutputs(ExecutionSequence executionSequence, float[] outputs)
        {
            executionSequence.EnqueueWriteBuffer(
                _layerOutputs,
                0,
                NodeCount,
                outputs
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