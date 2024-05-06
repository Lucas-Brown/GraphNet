package com.lucasbrown.NetworkTraining.DataSetTraining;

public class BetaDistribution implements ITrainableDistribution{

    private double alpha, beta, N;

    public BetaDistribution(double alpha, double beta, double N){
        this.alpha = alpha;
        this.beta = beta;
        this.N = N;
    }

    public double getAlpha()
    {
        return alpha;
    }

    public double getBeta(){
        return beta;
    }

    public double getSizeParameter(){
        return alpha + beta;
    }

    @Override 
    public double getNumberOfPointsInDistribution(){
        return N;
    }

    @Override
    public double getProbabilityDensity(double... x) {
        return getProbabilityDensity(x[0]);
    }

    public double getProbabilityDensity(double x){
        return densityOfPoint(x, alpha, beta);
    }

    public double getMean() {
        return alpha / (alpha + beta);
    }

    @Override
    public double getNormalizationConstant(){
        return normalizationConstant(alpha, beta);
    }
    
    @Override
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        double[] params = adjuster.getUpdatedParameters();
        alpha = params[0];
        alpha = params[1];
    }

    public static double normalizationConstant(double alpha, double beta){
        return Gamma.gamma(alpha)*Gamma.gamma(beta)/Gamma.gamma(alpha + beta);
    }

    public static double densityOfPoint(double x, double alpha, double beta){
        return Math.pow(x, alpha - 1)*Math.pow(1-x, beta - 1)/normalizationConstant(alpha, beta);
    }


}
