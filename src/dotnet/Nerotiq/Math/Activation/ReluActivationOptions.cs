namespace Nerotiq.Math.Activation {
    public class ReluActivationOptions : IActivationOptions
    {
        public IActivation Create()
        {
            return new ReluActivation();
        }
    }
}