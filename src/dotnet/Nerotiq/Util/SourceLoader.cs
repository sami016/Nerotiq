using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Nerotiq.Util
{
    public static class SourceLoader
    {
        private static readonly string[] SharedLibraryFiles = new []
        {
            "Nerotiq.shared.matrix.cl"
        };

        private static readonly string[] SharedLibraries = SharedLibraryFiles.Select(Read)
            .ToArray();

        /// <summary>
        /// Merged a set of source files ith the set of common libraries.
        /// </summary>
        public static string[] CreateProgramCollection(params string[] modules)
        {
            var output = new string[modules.Length + SharedLibraries.Length];
            for (var i=0; i<modules.Length; i++)
            {
                output[i] = modules[i];
            }
            for (var i=0; i<SharedLibraries.Length; i++)
            {
                output[i + modules.Length] = SharedLibraries[i];
            }
            return output;
        }


        public static string Read(string path) {

            using (var stream = typeof(SourceLoader).Assembly
                .GetManifestResourceStream(path)) 
            using (var reader = new StreamReader(stream))
            {
                return reader.ReadToEnd();
            }
        }

        /**
         * Concatenates several programs together.
         */
        public static string Concat(params string[] programs) {
            var stringBuilder = new StringBuilder();
            if (programs.Length > 0) {
                stringBuilder.Append(programs[0]);
            }
            for (var i=1; i<programs.Length; i++) 
            {
                stringBuilder.Append("\n\n");
                stringBuilder.Append(programs[i]);
            }
            return stringBuilder.ToString();
        }
    }
}