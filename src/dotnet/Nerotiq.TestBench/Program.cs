using System;
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

            program.Dispose();
        }
    }
}
