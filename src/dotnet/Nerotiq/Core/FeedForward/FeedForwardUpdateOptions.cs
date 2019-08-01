namespace Nerotiq.Core.FeedForward
{
    public class FeedForwardUpdateOptions : IOption<IFeedForwardUpdate>
    {
        public double LearningRate { get; set; } = 0.1;
        public IFeedForwardUpdate Create(ExecutionContext context)
        {
            return new FeedForwardUpdate(context, this);
        }
    }
}