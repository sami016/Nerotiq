using System;
using System.Runtime.InteropServices;

namespace Nerotiq.Util
{
    public static class MiscHelpers
    {
        static MiscHelpers() {
            IntPtrSize = Marshal.SizeOf(typeof(IntPtr));
            UIntPtrSize = Marshal.SizeOf(typeof(UIntPtr)) * 2;
        }

        public static readonly int IntPtrSize;
        public static readonly int UIntPtrSize;
    }
}