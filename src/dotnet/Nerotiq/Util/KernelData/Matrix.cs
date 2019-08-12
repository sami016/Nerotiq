using System;
using System.Collections.Generic;
using System.Text;

namespace Nerotiq.Util.KernelData
{
    /// <summary>
    /// Represents a matrix of size (n x m).
    /// </summary>
    public struct Matrix
    {
        public ushort Height;
        public ushort Width;
        public double[] Data;
    }
}
