
namespace Nerotiq.Data {

    /// <summary>
    /// Interface for a data source reader.
    /// </summary>
    public interface IDataSourceReader 
    {
        /// <summary>
        /// Reads the next data point, progressing the reader to the next position.
        /// </summary>
        /// <returns>data point</returns>
        double ReadNext();

        /// <summary>
        /// Gets the next data row.
        /// </summary>
        /// <returns>true is another row is available, else false</returns>
        bool GetNext();
    }
}