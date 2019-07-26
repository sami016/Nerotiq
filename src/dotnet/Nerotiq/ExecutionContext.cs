using System;
using Nerotiq.Exceptions;
using OpenCL.Net;

namespace Nerotiq {
    public class ExecutionContext {

        /**
         * The OpenCL context.
         */
        public Context OpenClContext { get; }

        /**
         * The active platform.
         */
        public Platform Platform { get; } 

        /**
         * The active device.
         */
        public Device Device { get; }

        /**
         * Default constructor to select first platform and device available.
         */
        public ExecutionContext(): this(
            NerotiqLib.EnumeratePlatforms()[0].Platform,
            NerotiqLib.EnumerateDevices(NerotiqLib.EnumeratePlatforms()[0].Platform)[0].Device
        ) {
        }

        /**
         * Create for a given platform and device.
         */
        public ExecutionContext(Platform platform, Device device)
        {
            Platform = platform;
            Device = device;

            OpenClContext = Cl.CreateContext(null, 1, new[] { Device }, null, IntPtr.Zero, out var error);
            if (error != ErrorCode.Success) {
                throw new NerotiqException($"Unable to create opencl context: {error}");
            }
        }
    }
}