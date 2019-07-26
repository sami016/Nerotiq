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
                Cl.WaitForEvents(1, new Event[] { _awaitEvent.Value });
                _awaitEvent = null;
            }
        }

        public void EnqueueWriteBuffer<T>(IMem<T> buffer, Bool blockingWrite, int offset, int length, T[] data) 
            where T : struct
        {
            Cl.EnqueueWriteBuffer(
                CommandQueue, 
                buffer, 
                blockingWrite, 
                offset, 
                length, 
                data,
                _awaitEvent == null ? 0 : 1, 
                _awaitEvent == null ? new Event[0] : new Event[]{ _awaitEvent.Value },
                out var awaitEvent
            );   
            _awaitEvent = awaitEvent;
        }
        
        public void EnqueueNDRangeKernel(Kernel kernel, uint workDim, IntPtr[] globalWorkOffset, IntPtr[] globalWorkSize, IntPtr[] localWorkSize) {
            Cl.EnqueueNDRangeKernel(
                CommandQueue,
                kernel, 
                workDim,
                globalWorkOffset, 
                globalWorkSize, 
                localWorkSize, 
                _awaitEvent == null ? (uint)0 : (uint)1, 
                _awaitEvent == null ? new Event[0] : new Event[]{ _awaitEvent.Value },
                out var awaitEvent
            );
            
        }
    }
}