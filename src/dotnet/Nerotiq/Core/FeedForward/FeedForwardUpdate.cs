using System;
using Nerotiq.Exceptions;
using Nerotiq.Math.Activation;
using Nerotiq.Util;
using Nerotiq.Util.Data;
using OpenCL.Net;

namespace Nerotiq.Core.FeedForward {

    /// <summary>
    /// Standard feed forard weight update.
    /// </summary>
    public class FeedForwardUpdate : IFeedForwardUpdate 
    {
        private static readonly string _source;
        private readonly FeedForwardUpdateOptions _options;
        private Program _program;
        private Kernel _updateKernel;
        private uint _dimensions;
        private int _nodeCount;

        static FeedForwardUpdate() {
            _source = SourceLoader.Read("Nerotiq.core.feedforward.update.cl");
        }

        public FeedForwardUpdate(ExecutionContext executionContext, FeedForwardUpdateOptions feedForwardUpdateOptions)
        {
            _options = feedForwardUpdateOptions;
            CompileKernels(executionContext);
        }

        private void CompileKernels(ExecutionContext executionContext) 
        {
            var sources = SourceLoader.CreateProgramCollection(_source);
            _program = Cl.CreateProgramWithSource(
                executionContext.OpenClContext,
                (uint)sources.Length, 
                sources,
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
            _updateKernel = Cl.CreateKernel(_program, "update", out error);
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error creating kernel update: {error}");
            }
        }

        public void SetUp(FeedForwardLayer layer, ILayer previousLayer)
        {
            _dimensions = (uint)layer.Dimensionality.Length;
            _nodeCount = layer.NodeCount;
            // Arg 0: learningRate (double)
            ClHelpers.SetKernelArg(
                _updateKernel,
                0,
                (double)_options.LearningRate
            );
            // Arg 1: previousLayerNodeCount (uint)
            ClHelpers.SetKernelArg(
                _updateKernel,
                1,
                (uint)previousLayer.NodeCount
            );
            // Arg 2: layerNodeCount (uint)
            ClHelpers.SetKernelArg(
                _updateKernel,
                2,
                (uint)layer.NodeCount
            );
            // Arg 3: layerDeltas (float*)
            layer.Deltas.SetKernelArg(
                _updateKernel,
                3
            );
            // Arg 4: layerWeights (float*)
            layer.Weights.SetKernelArg(
                _updateKernel,
                4
            );
            // Arg 5: layerBiases (float*)
            layer.Biases.SetKernelArg(
                _updateKernel,
                5
            );
            // Arg 6: previousLayerOutputs (float*)
            previousLayer.Outputs.SetKernelArg(
                _updateKernel,
                6
            );
        }

        public void Update(ExecutionSequence executionSequence) 
        {
            executionSequence.EnqueueNDRangeKernel(
                _updateKernel,
                _dimensions,
                null,
                new IntPtr [] { new IntPtr(_nodeCount) },
                null
            );
        }
    }
}