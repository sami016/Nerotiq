using OpenCL.Net;

namespace Nerotiq.Util {
    public class PlatformOption {
        public Platform Platform { get; }
        public string Name { get; }
        internal PlatformOption(Platform platform, string name) {
            Platform = platform;
            Name = name;
        }

    }
}