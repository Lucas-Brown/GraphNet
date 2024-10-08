package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.List;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.HelperClasses.WeightedDouble;
import com.lucasbrown.HelperClasses.WeightedDoubleCollector;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;

public class BetaDistributionAdjuster extends WeightedDoubleCollector implements Function {

    private static final double TOLLERANCE = 1E-6; // tollerance for optimization
    private static final int NM_ITTERATION_LIMIT = 1000;

    private double alpha, beta, N;

    private BetaDistribution distribution;

    public BetaDistributionAdjuster(ITrainableDistribution distribution) {
        this((BetaDistribution) distribution);
    }

    public BetaDistributionAdjuster(BetaDistribution distribution) {
        this.distribution = distribution;
        newPoints = new ArrayList<>();
    }

    public double getAlpha(){
        return alpha;
    }

    public double getBeta(){
        return beta;
    }

    public double getN(){
        return N;
    }

    @Override
    public double f(double... x) {
        variableTransform(x);

        double lambda_alpha = alpha * x[0];
        double lambda_beta = beta * x[1];
        
        double likelihood = N * logLikelihoodExpectationOfDistribution(lambda_alpha, lambda_beta);
        likelihood += logLikelihoodOfPoints(lambda_alpha, lambda_beta);
        return -likelihood; // maximize the likelihood -> minimize the negative of the likelihood
    }

    @Override
    public double f(Vec x) {
        return f(x.arrayCopy());
    }

    @Override
    public void prepareAdjustment(double weight, double value) {
        if (value < 0 || value > 1) {
            throw new InvalidParameterException("The beta distribution is bounded between 0 and 1");
        }
        super.prepareAdjustment(weight, value);
    }

    @Override
    public void applyAdjustments() {
        
        double N_points = WeightedDouble.getWeightSum(newPoints);
        if(N_points == 0){
            newPoints.clear();
            return; // make no changes
        }

        newPoints.forEach(wd -> wd.value = BetaDistribution.betaDataTransformation(wd.value, N_points));

        alpha = distribution.getAlpha();
        beta = distribution.getBeta();
        N = distribution.getNumberOfPointsInDistribution();

        NelderMead nm = new NelderMead();
        List<Vec> init_points = new ArrayList<>(3);

        init_points.add(new DenseVector(new double[] { -0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0, 0.2 }));

        double[] solution = nm.optimize(TOLLERANCE, NM_ITTERATION_LIMIT, this, init_points).arrayCopy();
        variableTransform(solution);
        alpha *= solution[0];
        beta *= solution[1];
        N += N_points;
        N = Math.min(N, GraphNetwork.N_MAX);

        newPoints.clear();
        distribution.applyAdjustments(this);
    }

    private void variableTransform(double[] vec) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = Math.exp(vec[i]);
        }
    }

    protected double logLikelihoodExpectationOfDistribution(double lambda_alpha, double lambda_beta) {
        return Math.log(BetaDistribution.normalizationConstant(alpha + lambda_alpha - 1, beta + lambda_beta - 1) /
                (BetaDistribution.normalizationConstant(alpha, beta)
                        * BetaDistribution.normalizationConstant(lambda_alpha, lambda_beta)));
    }

    private double logLikelihoodOfPoints(double lambda_alpha, double lambda_beta){
        return newPoints.stream()
            .filter(wp -> wp.weight > 0)
            .mapToDouble(wp -> wp.weight*BetaDistribution.densityOfPoint(wp.value, lambda_alpha, lambda_beta))
            .map(Math::log)
            .sum();
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[]{alpha, beta, N};
    }

}
