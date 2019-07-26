using System;
using System.Runtime.Serialization;

namespace Nerotiq.Exceptions
{
    public class NerotiqException : Exception
    {
        public NerotiqException()
        {
        }

        public NerotiqException(string message) : base(message)
        {
        }

        public NerotiqException(string message, Exception innerException) : base(message, innerException)
        {
        }

        protected NerotiqException(SerializationInfo info, StreamingContext context) : base(info, context)
        {
        }
    }
}