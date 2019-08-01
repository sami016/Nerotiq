namespace Nerotiq.Core 
{
    /// <summary>
    /// A generic option, acting as a factory for a given interface type.
    /// </summary>
    public interface IOption<T> {

        T Create(ExecutionContext context);

    }
}