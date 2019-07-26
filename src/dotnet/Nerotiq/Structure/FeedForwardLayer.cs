using System;
using Nerotiq.Exceptions;
using Nerotiq.Math.Activation;
using Nerotiq.Util;
using OpenCL.Net;

namespace Nerotiq.Structure {
    public class FeedForwardLayer : ILayer
    {
        private static readonly string _source;
        private readonly IMem<float> _layerSums;
        private readonly IMem<float> _layerOutputs;
        private readonly Kernel _kernel;
        private readonly Program _program;
        private readonly Kernel _forwardKernel;
        private readonly Kernel _backwardKernel;
        private readonly IActivation _activation;

        static FeedForwardLayer() {
            _source = SourceLoader.Read("Nerotiq.core.feedforward.cl");
        }

        public int NodeCount { get; set; }
        public ushort[] Dimensionality { get; set; }

        public FeedForwardLayer(ExecutionContext executionContext, FeedForwardLayerOptions options)
        {
            Dimensionality = options.Dimensionality;
            NodeCount = 1;
            foreach (var i in options.Dimensionality) {
                NodeCount *= i;
            }
            _activation = (options.ActivationOptions ?? new ReluActivationOptions())
                .Create();

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

            
            _layerSums = Cl.CreateBuffer<float>(
                executionContext.OpenClContext, 
                MemFlags.ReadWrite,
                NodeCount, 
                out error
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

        public void Dispose()
        {
            _program.Dispose();
            _forwardKernel.Dispose();
            _backwardKernel.Dispose();
            _layerSums?.Dispose();
            _layerOutputs?.Dispose();
        }
    }
}