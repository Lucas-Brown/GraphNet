package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.List;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;

public class BetaDistributionAdjuster implements Function, IExpectationAdjuster {

    private static final double TOLLERANCE = 1E-6; // tollerance for optimization
    private static final int NM_ITTERATION_LIMIT = 1000;

    private double alpha, beta, N;

    private BetaDistribution distribution;

    private ArrayList<WeightedDouble> newPoints;

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

    public void prepareAdjustment(double weight, double value) {
        if (value < 0 || value > 1) {
            throw new InvalidParameterException("The beta distribution is bounded between 0 and 1");
        }
        newPoints.add(new WeightedDouble(weight, value));
    }

    @Override
    public void prepareAdjustment(double weight, double[] newPoint) {
        if(newPoint.length > 1){
            throw new InvalidParameterException("This distribution only accepts a single degree of input. ");
        }
        prepareAdjustment(weight, newPoint[0]);
    }

    @Override
    public void prepareAdjustment(double[] newData) {
        prepareAdjustment(1, newData);
    }

    @Override
    public void applyAdjustments() {
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
        N += newPoints.stream().mapToDouble(wp -> wp.weight).sum();

        newPoints.clear();
        distribution.applyAdjustments(this);
    }

    private void variableTransform(double[] vec) {
        for (int i = 0; i < vec.length; i++) {
            vec[i] = Math.exp(vec[i]);
        }
    }

    private double logLikelihoodExpectationOfDistribution(double lambda_alpha, double lambda_beta) {
        return Math.log(BetaDistribution.normalizationConstant(alpha + lambda_alpha - 1, beta + lambda_beta - 1) /
                (BetaDistribution.normalizationConstant(alpha, beta)
                        * BetaDistribution.normalizationConstant(lambda_alpha, lambda_beta)));
    }

    private double logLikelihoodOfPoints(double lambda_alpha, double lambda_beta){
        return newPoints.stream()
            .mapToDouble(wp -> wp.weight*BetaDistribution.densityOfPoint(wp.value, lambda_alpha, lambda_beta))
            .map(Math::log)
            .sum();
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[]{alpha, beta, N};
    }

}
