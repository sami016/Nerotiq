using System;
using Nerotiq.Exceptions;
using OpenCL.Net;

namespace Nerotiq
{
    /**
     * A single runner for execution containing a single opencl command queue.
     */
    public class ExecutionSequence
    {
        public CommandQueue CommandQueue { get; }
        public ExecutionContext Context { get; }

        private Event? _awaitEvent = null;
        
        public ExecutionSequence(ExecutionContext context)
        {
            Context = context;
            CommandQueue = Cl.CreateCommandQueue(context.OpenClContext, context.Device, CommandQueueProperties.None, out var error);
            var errorCode = error;
            if (errorCode != ErrorCode.Success) {
                throw new NerotiqException($"Unable to create opencl command queue: {errorCode}");
            }
        }

        /**
         * Blocks until the last event has finished execution.
         */
        public void FinishExecution() {
            if (_awaitEvent.HasValue) {
                var errorCode = Cl.WaitForEvents(1, new Event[] { _awaitEvent.Value });
                _awaitEvent = null;
                if (errorCode != ErrorCode.Success) {
                    throw new Exception($"Failed to await events: {errorCode}");
                }
            }
        }

        public void EnqueueReadBuffer<T>(IMem<T> buffer, int offset, int length, T[] data) 
            where T : struct
        {
            Cl.EnqueueReadBuffer(
                CommandQueue,
                buffer,
                Bool.True,
                offset,
                length,
                data,
                0,//_awaitEvent == null ? 0 : 1, 
                null,//_awaitEvent == null ? new Event[0] : new Event[]{ _awaitEvent.Value },
                out var awaitEvent
            );
            //_awaitEvent = awaitEvent;
        }

        public T[] ReadBuffer<T>(IMem<T> buffer, int offset, int length) 
            where T : struct
        {
            var data = new T[length];
            Cl.EnqueueReadBuffer(
                CommandQueue,
                buffer,
                Bool.True,
                offset,
                length,
                data,
                0,//_awaitEvent == null ? 0 : 1, 
                null,//_awaitEvent == null ? new Event[0] : new Event[]{ _awaitEvent.Value },
                out var awaitEvent
            );
            //s_awaitEvent = awaitEvent;
            return data;
        }

        public void EnqueueWriteBuffer<T>(IMem<T> buffer, int offset, int length, T[] data) 
            where T : struct
        {
            Cl.EnqueueWriteBuffer(
                CommandQueue, 
                buffer, 
                Bool.True, 
                offset, 
                length, 
                data,
                0,//_awaitEvent == null ? 0 : 1, 
                null,//_awaitEvent == null ? new Event[0] : new Event[]{ _awaitEvent.Value },
                out var awaitEvent
            );   
            //_awaitEvent = awaitEvent;
        }
        
        public void EnqueueNDRangeKernel(Kernel kernel, uint workDim, IntPtr[] globalWorkOffset, IntPtr[] globalWorkSize, IntPtr[] localWorkSize) {
            var errorCode = Cl.EnqueueNDRangeKernel(
                CommandQueue,
                kernel, 
                workDim,
                globalWorkOffset, 
                globalWorkSize, 
                localWorkSize, 
                _awaitEvent == null ? (uint)0 : (uint)1, 
                _awaitEvent == null ? null : new Event[]{ _awaitEvent.Value },
                out var awaitEvent
            );
            if (errorCode != ErrorCode.Success)
            {
                throw new NerotiqException($"Error enqueueing ND range kernel: {errorCode}");
            }
            _awaitEvent = awaitEvent;
        }
    }
}