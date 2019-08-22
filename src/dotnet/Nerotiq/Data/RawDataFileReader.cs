
using System.IO;

namespace Nerotiq.Data {

    /// <summary>
    /// Reads in data from a raw data file.
    /// Such files follow a row structure, with an initial header defining the number of samples per row and the number of rows.     
    /// </summary>
    public class RawDataFileReader : IDataSourceReader 
    {
        private readonly FileStream _fileStream;
        private readonly BinaryReader _binaryReader;
        private readonly ushort _rowWidth;
        private readonly ulong _totalRows;

        private uint _currentColumnPosition = 0;
        private uint _currentRowPosition = 0;
        public RawDataFileReader(string filePath)
        {
            if (!File.Exists(filePath)) 
            {
                throw new IOException($"file '{filePath}' did not exist");
            }
            _fileStream = File.Open(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            _binaryReader = new BinaryReader(_fileStream);

            _rowWidth = _binaryReader.ReadUInt16();
            _totalRows = _binaryReader.ReadUInt64();
        }

        public double ReadNext() 
        {
            return _binaryReader.ReadDouble();
        }

        public bool GetNext()
        {
            _currentColumnPosition = 0;
            _currentRowPosition++;

            return _currentRowPosition != _totalRows;
        }
    }
}