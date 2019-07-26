using System.IO;
using System.Text;

namespace Nerotiq.Util
{
    public static class SourceLoader
    {
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