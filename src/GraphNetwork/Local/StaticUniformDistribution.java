package src.GraphNetwork.Local;

public class StaticUniformDistribution extends ActivationProbabilityDistribution 
{

    @Override
    public boolean shouldSend(double inputSignal, double factor) {
        return true;
    }

    @Override
    public void reinforceDistribution(double valueToReinforce) {
        // do nothing
    }

    @Override
    public void diminishDistribution(double valueToDiminish) {
        // do nothing
    }

    @Override
    public double getDirectionOfDecreasingLikelyhood(double x) {
        return 0;
    }
    
}
