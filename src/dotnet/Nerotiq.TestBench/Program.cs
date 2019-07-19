using System;
using System.Runtime.InteropServices;
using OpenCL.Net;
using OpenCL.Net.Extensions;

namespace Nerotiq.TestBench
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            uint platformCount;
            ErrorCode result = Cl.GetPlatformIDs(0, null, out platformCount);
            Console.WriteLine("{0} platforms found", platformCount);
            

            var platformIds = new Platform[platformCount];
            result = Cl.GetPlatformIDs(platformCount, platformIds, out platformCount);
            var platformCounter = 0;
            foreach (var platformId in platformIds) {
                IntPtr paramSize;
                result = Cl.GetPlatformInfo(platformId, PlatformInfo.Name, IntPtr.Zero, InfoBuffer.Empty, out paramSize);

                using (var buffer = new InfoBuffer(paramSize))
                {
                    result = Cl.GetPlatformInfo(platformIds[0], PlatformInfo.Name, paramSize, buffer, out paramSize);

                    Console.WriteLine($"Platform {platformCounter}: {buffer}");
                }
                platformCounter++;
            }

            Console.WriteLine($"Using first platform...");
            
            uint deviceCount;
            result = Cl.GetDeviceIDs(platformIds[0], DeviceType.All, 0, null, out deviceCount);
            Console.WriteLine("{0} devices found", deviceCount);

            var deviceIds = new Device[deviceCount];
            result = Cl.GetDeviceIDs(platformIds[0], DeviceType.All, deviceCount, deviceIds, out var numberDevices);
            
            var selectedDevice = deviceIds[0];

            var context = Cl.CreateContext(null, 1, new[] { selectedDevice }, null, IntPtr.Zero, out var error);

            const string correctSource = @"
            // Simple test; c[i] = a[i] + b[i]
            __kernel void add_array(__global float *a, __global float *b, __global float *c)
            {
                int xid = get_global_id(0);
                c[xid] = a[xid] + b[xid];
            }
            
            __kernel void sub_array(__global float *a, __global float *b, __global float *c)
            {
                int xid = get_global_id(0);
                c[xid] = a[xid] - b[xid];
            }
            ";

                
            var program = Cl.CreateProgramWithSource(context, 1, new[] { correctSource }, null, out var error2);
            error2 = Cl.BuildProgram(program, 1, new[] { selectedDevice }, string.Empty, null, IntPtr.Zero);
    
            Console.WriteLine(error2);

            // Get the kernels.
            var kernels = Cl.CreateKernelsInProgram(program, out error);
            Console.WriteLine($"Program contains {kernels.Length} kernels.");
            var kernel = kernels[0];

            //
            float[] A = new float[1000];
            float[] B = new float[1000];
            float[] C = new float[1000];

            for (var i=0; i<1000; i++) {
                A[i] = i;
                B[i] = i;
            }

            IMem<float> hDeviceMemA = Cl.CreateBuffer(context, MemFlags.CopyHostPtr | MemFlags.ReadOnly, A, out error);
            IMem<float> hDeviceMemB = Cl.CreateBuffer(context, MemFlags.CopyHostPtr | MemFlags.ReadOnly, B, out error);
            IMem<float> hDeviceMemC = Cl.CreateBuffer(context, MemFlags.CopyHostPtr | MemFlags.ReadOnly, C, out error);

            // Create a command queue.
            var cmdQueue = Cl.CreateCommandQueue(context, selectedDevice, CommandQueueProperties.None, out error);

            
            int intPtrSize = 0;
            intPtrSize = Marshal.SizeOf(typeof(IntPtr));

            error = Cl.SetKernelArg(kernel, 0, new IntPtr(intPtrSize), hDeviceMemA);
            error = Cl.SetKernelArg(kernel, 1, new IntPtr(intPtrSize), hDeviceMemB);
            error = Cl.SetKernelArg(kernel, 2, new IntPtr(intPtrSize), hDeviceMemC);


            // write data from host to device
            Event clevent;
            error = Cl.EnqueueWriteBuffer(cmdQueue, hDeviceMemA, Bool.True, IntPtr.Zero,
                new IntPtr(1000 * sizeof(float)),
                A, 0, null, out clevent);
            error = Cl.EnqueueWriteBuffer(cmdQueue, hDeviceMemB, Bool.True, IntPtr.Zero,
                new IntPtr(1000 * sizeof(float)),
                B, 0, null, out clevent);

            // execute kernel
            error = Cl.EnqueueNDRangeKernel(cmdQueue, kernel, 1, null, new IntPtr[] { new IntPtr(1000) }, null, 0, null, out clevent);
            Console.WriteLine($"Run result: {error}");
            
            

            error = Cl.EnqueueReadBuffer(cmdQueue, hDeviceMemC, Bool.True, 0, C.Length, C, 0, null, out clevent);

            for (var i=0; i<1000; i++) {
                Console.WriteLine($"[{i}]: {C[i]}");
            }
            
            program.Dispose();
        }
    }
}
