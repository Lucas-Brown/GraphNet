package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.ActivationProbabilityDistribution;

import jsat.math.integration.Romberg;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;
import jsat.linear.Vec;
import jsat.linear.vectorcollection.VectorArray;

public class Convolution {

    private final Random rng = new Random();

    private ArrayList<ActivationProbabilityDistribution> activationDistributions;
    private ArrayList<ActivationFunction> activators;
    private double[] weights;
    private List<DoubleUnaryOperator> distributions;
    private DoubleUnaryOperator conv_func;

    public Convolution(ArrayList<ActivationProbabilityDistribution> activationDistributions,
            ArrayList<ActivationFunction> activators, double[] weights) {
        this.activationDistributions = activationDistributions;
        this.activators = activators;
        this.weights = weights;

        // combine distributions, activation functions, and weights into one probability
        // distribution
        distributions = IntStream.range(0, activationDistributions.size())
                .mapToObj(i -> applyActivationToDistribution(activationDistributions.get(i), activators.get(i),
                        weights[i]))
                .toList();

        // combine distribution functions into a single convolution function
        conv_func = distributions.get(0);
        for (int i = 1; i < distributions.size(); i++) {
            conv_func = convolution(conv_func, distributions.get(i));
        }
    }

    /**
     * Get the convolution of a collection of distributions with correspoding
     * activation functions and weights for a given point.
     * 
     * This is an INCREDIBLY slow computation that scales poorly with the number of
     * distributions. May need to consider using fourier transforms for N > 3?
     * 
     * @param activationDistributions
     * @param activator
     * @param weights
     * @param z
     * @return
     */
    public double convolve(double z) {
        // Compute the convolution function at z
        return conv_func.applyAsDouble(z);
    }

    /**
     * Sample the convolution for a value such that x1 + x2 ... = z
     * 
     * @param z
     * @return
     */
    public double[] sample(double z) {
        // Getting a true sample is far too difficult and probably not worth it
        // Assume that each distribution is approximately normal

        double[] means = IntStream.range(0, activationDistributions.size())
                .mapToDouble(i -> activationDistributions.get(i).getMeanOfAppliedActivation(activators.get(i)))
                .toArray();
        double[] vars = IntStream.range(0, activationDistributions.size())
                .mapToDouble(
                        i -> activationDistributions.get(i).getVarianceOfAppliedActivation(activators.get(i), means[i]))
                .toArray();
        
        // initialize the covariance matrix by setting the values of it's inverse 
        // this is an odd processes that mimics the idea of setting a variable to 
        // a constant by removing its corresponding row and collumn while maintaining 
        // its interference on the other variables 
        Matrix covarianceMatrix = new DenseMatrix(vars.length-1, vars.length-1);
        double var2_N = vars[vars.length-1] * vars[vars.length-1];
        for(int i = 0; i < vars.length-1; i++)
        {
            covarianceMatrix.set(i, i, 1/(vars[i] * vars[i]) + 1/var2_N);
            for(int j = 0; j < i; j++)
            {
                covarianceMatrix.set(i, j, 1/var2_N);
                covarianceMatrix.set(j, i, 1/var2_N);
            }
        }

        // Use the LUP decomposition to invert the covariance matrix
        Matrix[] LUP = covarianceMatrix.lup();
        LUPDecomposition decomp = new LUPDecomposition(LUP[0],LUP[1],LUP[2]);
        covarianceMatrix = decomp.solve(Matrix.eye(vars.length)); // inverse

        // create the multivariate normal distribution
        NormalM norm = new NormalM();
        norm.setMeanCovariance(new DenseVector(means), covarianceMatrix);

        // get a single sample
        double[] sample = norm.sample(1, rng).get(0).arrayCopy();
        
        // add the final conditional sample
        System.arraycopy(new double[sample.length + 1], 0, sample, 0, sample.length);
        sample[sample.length-1] = z - DoubleStream.of(sample).sum();

        return sample;
    }

    /**
     * Convolution of two functions
     * 
     * @param f1
     * @param f2
     * @return
     */
    private static DoubleUnaryOperator convolution(DoubleUnaryOperator f1, DoubleUnaryOperator f2) {
        return z -> Romberg.romb(new DoubleFunction(convolutionIntegrand(f1, f2, z)), -1, 1);
    }

    private static DoubleUnaryOperator convolutionIntegrand(DoubleUnaryOperator f1, DoubleUnaryOperator f2, double z) {
        return t -> IntegralTransformations
                .hyperbolicTangentTransform(x -> f1.applyAsDouble(x) * f2.applyAsDouble(z - x), t);
    }

    private static DoubleUnaryOperator applyActivationToDistribution(
            ActivationProbabilityDistribution activationDistribution, ActivationFunction activator, double weight) {
        return x -> activationDistribution.getProbabilityDensity(activator.inverse(x) / weight)
                * activator.inverseDerivative(x) / weight;
    }

}
