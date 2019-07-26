using System;
using System.Collections.Generic;
using Nerotiq.Util;
using OpenCL.Net;

namespace Nerotiq
{
    /**
     * Static entrypoint class for Nerotiq library. 
     **/
    public static class NerotiqLib
    {
        public static IList<PlatformOption> EnumeratePlatforms() {
            uint platformCount;
            ErrorCode result = Cl.GetPlatformIDs(0, null, out platformCount);
            // Load platform ids.
            var platforms = new Platform[platformCount];
            result = Cl.GetPlatformIDs(platformCount, platforms, out platformCount);
            var platformList = new List<PlatformOption>();
            foreach (var platform in platforms) {
                IntPtr paramSize;
                result = Cl.GetPlatformInfo(platform, PlatformInfo.Name, IntPtr.Zero, InfoBuffer.Empty, out paramSize);

                string name = "";
                using (var buffer = new InfoBuffer(paramSize))
                {
                    result = Cl.GetPlatformInfo(platform, PlatformInfo.Name, paramSize, buffer, out paramSize);
                    name = buffer.ToString();
                }

                platformList.Add(new PlatformOption(platform, name));
            }
            return platformList;
        }

        public static IList<DeviceOption> EnumerateDevices(Platform platform, DeviceType deviceType = DeviceType.All) {
            uint deviceCount;
            var result = Cl.GetDeviceIDs(platform, deviceType, 0, null, out deviceCount);
            var devices = new Device[deviceCount];
            result = Cl.GetDeviceIDs(platform, deviceType, deviceCount, devices, out var numberDevices);
            
            var deviceList = new List<DeviceOption>();
            foreach (var device in devices) {
                IntPtr paramSize;
                result = Cl.GetDeviceInfo(device, DeviceInfo.Name, IntPtr.Zero, InfoBuffer.Empty, out paramSize);

                string name = "";
                using (var buffer = new InfoBuffer(paramSize))
                {
                    result = Cl.GetDeviceInfo(device, DeviceInfo.Name, paramSize, buffer, out paramSize);
                    name = buffer.ToString();
                }

                deviceList.Add(new DeviceOption(device, name));
            }
            return deviceList;
        }

    }
}