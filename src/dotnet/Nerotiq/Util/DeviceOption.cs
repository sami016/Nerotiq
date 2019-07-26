using OpenCL.Net;

namespace Nerotiq.Util
{
    public class DeviceOption
    {
        public Device Device { get; }
        public string Name { get; }
        public DeviceOption(Device device, string name) {
            Device = device;
            Name = name;
        }
    }
}