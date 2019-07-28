using Nerotiq.Exceptions;
using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Text;

namespace Nerotiq.Util
{
    public static class ClHelpers
    {
        public static void SetKernelArg(Kernel kernel, uint argIndex, IntPtr argSize, object argValue)
        {
            var errorCode = Cl.SetKernelArg(
                kernel,
                argIndex,
                argSize,
                argValue
            );
            if (errorCode != ErrorCode.Success)
            {
                throw new NerotiqException($"Error setting kernel arg at index {argIndex}: {errorCode}");
            }
        }
        public static void SetKernelArg<T>(Kernel kernel, uint argIndex, T argValue)
            where T : struct
        {
            var errorCode = Cl.SetKernelArg(kernel, argIndex, argValue);
            if (errorCode != ErrorCode.Success)
            {
                throw new NerotiqException($"Error setting kernel arg at index {argIndex}: {errorCode}");
            }
        }
    }
}
