package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.function.Function;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.IConvolution;

public class BetaDistribution implements ITrainableDistribution{

    private double alpha, beta, N;

    public BetaDistribution(double alpha, double beta){
        this(alpha, beta, 10);
    }

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
        beta = params[1];
        N = params[2];
    }

    public static double normalizationConstant(double alpha, double beta){
        return Gamma.gamma(alpha)*Gamma.gamma(beta)/Gamma.gamma(alpha + beta);
    }

    public static double densityOfPoint(double x, double alpha, double beta){
        return Math.pow(x, alpha - 1)*Math.pow(1-x, beta - 1)/normalizationConstant(alpha, beta);
    }

    @Override
    public Function<ITrainableDistribution, IExpectationAdjuster> getDefaulAdjuster() {
        return BetaDistributionAdjuster::new;
    }

    @Override
    public IConvolution toConvolution(ActivationFunction activator, double weight) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'toConvolution'");
    }

    public static BetaDistribution getUniformBetaDistribution()
    {
        return new BetaDistribution(1, 1);
    }


}
