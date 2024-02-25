package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.Arrays;
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
     * @param z
     * @return
     */
    public double convolve(double z) {
        // Compute the convolution function at z
        return conv_func.applyAsDouble(z);
    }

    /**
     * Get a single sample of the convolution such that x1 + x2 ... = z
     * @param z
     * @return
     */
    public double[] sample(double z) {
        return sample(z, 1)[0];
    }

    /**
     * Sample the convolution for a value such that x1 + x2 ... = z
     * 
     * @param z
     * @param n the number of samples
     * @return
     */
    public double[][] sample(double z, int n) {
        // if there's only 1 distribution, then there's zero degrees of freedom
        // return the input value
        if (activationDistributions.size() <= 1) {
            double[][] sample = new double[n][1];
            for (int i = 0; i < sample.length; i++) {
                sample[i][0] = z;
            }
            return sample;
        }

        // Getting a true sample is far too difficult and probably not worth it
        // Assume that each distribution is approximately normal

        // Get the mean and variance of each distribution
        double[] means = IntStream.range(0, activationDistributions.size())
                .mapToDouble(
                        i -> activationDistributions.get(i).getMeanOfAppliedActivation(activators.get(i), weights[i]))
                .toArray();
        double[] vars = IntStream.range(0, activationDistributions.size())
                .mapToDouble(
                        i -> activationDistributions.get(i).getVarianceOfAppliedActivation(activators.get(i),
                                weights[i], means[i]))
                .toArray();

        double[] vars2 = DoubleStream.of(vars).map(s -> s*s).toArray();

        // The rest of this procedure follows this stack exchange post 
        // https://stats.stackexchange.com/questions/617653/how-can-i-sample-a-multivariate-normal-vector-that-satisfies-a-linear-equality-c

        double mean_sum = DoubleStream.of(means).sum();
        double var2_sum = DoubleStream.of(vars2).sum();

        double z_shifted = z - mean_sum;

        // construct the conditional mean vector
        double[] mean_conditional = new double[means.length - 1];
        for (int i = 0; i < mean_conditional.length; i++) {
            mean_conditional[i] = means[i] + vars2[i]*z_shifted/var2_sum;
        }

        // construct the conditional variance matrix
        Matrix variance_conditional = new DenseMatrix(vars.length - 1, vars.length - 1);

        for (int i = 0; i < variance_conditional.cols(); i++) {
            // diagonal components
            variance_conditional.set(i, i, vars2[i] * (1 - vars2[i]/var2_sum));
            
            // off-diagonal components
            for (int j = 0; j < i; j++) {
                double covariance = -vars2[i]*vars2[j]/var2_sum;
                variance_conditional.set(i, j, covariance);
                variance_conditional.set(j, i, covariance);
            }
        }

        // create the multivariate normal distribution
        NormalM norm = new NormalM(new DenseVector(mean_conditional), variance_conditional);

        // get samples
        List<Vec> samples_conditional = norm.sample(n, rng);

        // use the sample to generate the remaining component
        double[][] samples = new double[n][means.length];
        for (int i = 0; i < samples.length; i++) {
            double[] sample_i = samples_conditional.get(i).arrayCopy();
            System.arraycopy(sample_i, 0, samples[i], 0, sample_i.length);
            samples[i][samples[i].length-1] = z - DoubleStream.of(sample_i).sum();
        }

        return samples;
    }

    /**
     * Convolution of two functions
     * 
     * @param f1
     * @param f2
     * @return
     */
    public static DoubleUnaryOperator convolution(DoubleUnaryOperator f1, DoubleUnaryOperator f2) {
        return z -> Romberg.romb(new DoubleFunction(convolutionIntegrand(f1, f2, z)), -1, 1);
    }

    public static DoubleUnaryOperator convolutionIntegrand(DoubleUnaryOperator f1, DoubleUnaryOperator f2, double z) {
        return t -> IntegralTransformations
                .asymptoticTransform(x -> f1.applyAsDouble(x) * f2.applyAsDouble(z - x), t);
    }

    public static DoubleUnaryOperator applyActivationToDistribution(
            ActivationProbabilityDistribution activationDistribution, ActivationFunction activator, double weight) {
        return x -> activationDistribution.getProbabilityDensity(activator.inverse(x) / weight)
                * Math.abs(activator.inverseDerivative(x) / weight);
    }

}
