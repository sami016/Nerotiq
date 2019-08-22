using System;
using OpenCL.Net;

namespace Nerotiq.Util.Data
{
    /// <summary>
    /// Represents a resource that is stored upon the gpu.
    /// </summary>
    public interface IGpuResource
    {
        /// <summary>
        /// Assigns the resource to a kernel argument.
        /// </summary>
        /// <param name="kernel">kernel</param>
        /// <param name="argIndex">arg index</param>
        void SetKernelArg(Kernel kernel, uint argIndex);

        /// <summary>
        /// Reads the data into memory temporarily.
        /// This memory will be freed upon disposal.
        /// </summary>
        /// <param name="executionSequence">execution sequence</param>
        /// <returns>dispoable</returns>
        IDisposable Read(ExecutionSequence executionSequence);
    }
}