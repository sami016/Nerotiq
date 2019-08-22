using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using Nerotiq.Exceptions;
using OpenCL.Net;

namespace Nerotiq.Util.Data
{
    public class GpuMatrix : IGpuResource, IDisposable
    {
        public ushort Width { get; }
        public ushort Height { get; }
        public IMem<double> Data { get; }

        /// <summary>
        /// A copy of the data in memory.
        /// Can only be accessed within a read block.
        /// </summary>
        /// <value></value>
        public double[] InMemoryData { get; private set; }

        public GpuMatrix(ushort rows, ushort columns, ExecutionContext context) 
        {
            Width = columns;
            Height = rows;
            Data = Cl.CreateBuffer<double>(
                context.OpenClContext, 
                MemFlags.ReadWrite,
                Width * Height * sizeof(double), 
                out var error
            );
            if (error != ErrorCode.Success) 
            {
                throw new NerotiqException($"Error creating matrix buffer: {error}");
            }
        }

        public void Dispose() {
            Data?.Dispose();
        }

        public void Update(double[] weights, ExecutionSequence executionSequence)
        {
            executionSequence.EnqueueWriteBuffer<double>(
                Data,
                0,
                weights.Length,
                weights
            );
        }
        
        public void Update(ExecutionSequence executionSequence)
        {
            if (InMemoryData == null) 
            {
                throw new InvalidOperationException("Read not called");
            }
            executionSequence.EnqueueWriteBuffer<double>(
                Data,
                0,
                InMemoryData.Length,
                InMemoryData
            );
        }

        public IDisposable Read(ExecutionSequence executionSequence) 
        {
            // Read the data into memory.
            InMemoryData = executionSequence.ReadBuffer(
                Data,
                0,
                Width * Height
            );
            return new ActionDisposable(() => {
                // Release the memory for garbage collection when we're done with it.
                InMemoryData = null;
            });
        }

        public double GetValue(int row, int col) 
        {
            if (InMemoryData == null) 
            {
                throw new InvalidOperationException("Read not called");
            }
            return InMemoryData[row * Width + col];
        }

        public void SetValue(int row, int col, double value) 
        {
            if (InMemoryData == null) 
            {
                throw new InvalidOperationException("Read not called");
            }
            InMemoryData[row * Width + col] = value;
        }

        public void SetKernelArg(Kernel kernel, uint argIndex)
        {
            ClHelpers.SetKernelArg(
                kernel,
                argIndex,
                new IntPtr(MiscHelpers.IntPtrSize),
                Data
            );
        }

        public static void SetNullKernelArg(Kernel kernel, uint argIndex) {
            ClHelpers.SetKernelArg(
                kernel,
                argIndex,
                new IntPtr(0)
            );
        }
    }
}
