using System;

namespace Nerotiq.IO
{
    public class SerializationNameAttribute : Attribute
    {
        public string Name { get; }

        public SerializationNameAttribute(string name)
        {
            this.Name = name;
        }
    }
}