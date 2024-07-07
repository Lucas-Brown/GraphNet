package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.util.function.Function;

public class BernoulliDistribution implements ITrainableDistribution {

    private double p, N;

    public BernoulliDistribution(double p, double N){
        this.p = p;
        this.N = N;
    }
    
    public double getP() {
        return p;
    }

    @Override
    public double getProbabilityDensity(double... x) {
        return getProbabilityDensity(x);
    }

    public double getProbabilityDensity(double x){
        if(x == 1){
            return p; 
        }
        else if(x == 0){
            return 1-p;
        }
        else{
            return Math.pow(p, x)*Math.pow(1-p, 1-x);
        }
    }

    @Override
    public double getNumberOfPointsInDistribution() {
        return N;
    }

    @Override
    public double getNormalizationConstant() {
        return 1;
    }

    @Override
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        double[] params = adjuster.getUpdatedParameters();
        p = params[0];
        N = params[1];
    }


    @Override
    public Function<ITrainableDistribution, IExpectationAdjuster> getDefaulAdjuster() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getDefaulAdjuster'");
    }

    public static BernoulliDistribution getEvenDistribution(){
        return new BernoulliDistribution(0.5, 10);
    }
    
}
