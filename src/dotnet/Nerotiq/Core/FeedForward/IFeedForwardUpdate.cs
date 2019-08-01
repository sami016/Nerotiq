using Nerotiq.Math.Activation;

namespace Nerotiq.Core.FeedForward {
    public interface IFeedForwardUpdate 
    {
        void SetUp(FeedForwardLayer layer, ILayer previousLayer);
        void Update(ExecutionSequence executionSequence);
    }
}