package com.lucasbrown.NetworkTraining.ApproximationTools;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;

import jsat.math.integration.Romberg;
import jsat.distributions.multivariate.NormalM;
import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Matrix;
import jsat.linear.Vec;

public class Convolution {

    private final Random rng = new Random();

    private ArrayList<FilterDistribution> activationDistributions;
    private ArrayList<ActivationFunction> activators;
    private double[] weights;
    private List<DoubleUnaryOperator> distributions;
    private DoubleUnaryOperator conv_func;

    private int[] dependentIndices;
    private int[] independentIndices;

    public Convolution(ArrayList<FilterDistribution> activationDistributions,
            ArrayList<ActivationFunction> activators, double[] weights) {
        this.activationDistributions = activationDistributions;
        this.activators = activators;
        this.weights = weights;

        // If the weight of a distribution is exactly 0, then that distribution does not
        // contribute to the convolution
        dependentIndices = IntStream.range(0, weights.length).filter(i -> weights[i] != 0).toArray();
        independentIndices = IntStream.range(0, weights.length).filter(i -> weights[i] == 0).toArray();

        // combine distributions, activation functions, and weights into one probability
        // distribution
        distributions = IntStream.range(0, activationDistributions.size())
                .mapToObj(i -> applyActivationToDistribution(activationDistributions.get(i), activators.get(i),
                        weights[i]))
                .toList();

        if (dependentIndices.length == 0)
            return;

        // combine dependent distribution functions into a single convolution function
        conv_func = distributions.get(dependentIndices[0]);
        for (int i = 1; i < dependentIndices.length; i++) {
            int idx = dependentIndices[i];
            conv_func = convolution(conv_func, distributions.get(idx));
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
        if (dependentIndices.length == 0) {
            return 0;
        } else {
            // Compute the convolution function at z
            return conv_func.applyAsDouble(z);
        }

    }

    /**
     * Get a single sample of the convolution such that x1 + x2 ... = z
     * 
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
     * @param count the number of samples
     * @return
     */
    public double[][] sample(double z, int count) {
        double[][] independent_samples = generateIndependentSamples(count);
        double[][] dependent_samples = generateDependentSamples(z, count);

        return mergeSamples(dependent_samples, independent_samples);
    }

    private double[][] generateIndependentSamples(int count) {
        double[][] independent_samples = new double[count][independentIndices.length];

        for (int i = 0; i < independentIndices.length; i++) {
            int idx = independentIndices[i];
            FilterDistribution dist = activationDistributions.get(idx);
            ActivationFunction af = activators.get(idx);
            double w = weights[idx];

            for (int n = 0; n < count; n++) {
                independent_samples[n][i] = w * af.activator(dist.sample());
            }
        }
        return independent_samples;
    }

    private double[][] generateDependentSamples(double z, int count) {
        final int dof = dependentIndices.length - 1;

        // if there's only 1 dependent sample then there's no degrees of freedom (i.e,
        // return z)
        if (dependentIndices.length == 0) {
            return new double[count][0];
        } else if (dependentIndices.length == 1) {
            double[][] sample = new double[count][1];
            for (int n = 0; n < sample.length; n++) {
                sample[n] = new double[] { z };
            }
            return sample;
        }

        // Getting a true sample is far too difficult and probably not worth it
        // Assume that each distribution is approximately normal

        // Get the mean and variance of each distribution
        double[] means = IntStream.of(dependentIndices)
                .mapToDouble(
                        i -> activationDistributions.get(i).getMeanOfAppliedActivation(activators.get(i), weights[i]))
                .toArray();
        double[] vars = IntStream.of(dependentIndices)
                .mapToDouble(
                        i -> activationDistributions.get(i).getVarianceOfAppliedActivation(activators.get(i),
                                weights[i], means[i]))
                .toArray();

        double[] vars2 = DoubleStream.of(vars).map(s -> s * s).toArray();

        // The rest of this procedure follows this stack exchange post
        // https://stats.stackexchange.com/questions/617653/how-can-i-sample-a-multivariate-normal-vector-that-satisfies-a-linear-equality-c

        double mean_sum = DoubleStream.of(means).sum();
        double var2_sum = DoubleStream.of(vars2).sum();

        double z_shifted = z - mean_sum;

        // construct the conditional mean vector
        double[] mean_conditional = new double[dof];
        for (int i = 0; i < dof; i++) {
            mean_conditional[i] = means[i] + vars2[i] * z_shifted / var2_sum;
        }

        // construct the conditional variance matrix
        Matrix variance_conditional = new DenseMatrix(dof, dof);

        for (int i = 0; i < dof; i++) {
            // diagonal components
            variance_conditional.set(i, i, vars2[i] * (1 - vars2[i] / var2_sum));

            // off-diagonal components
            for (int j = 0; j < i; j++) {
                double covariance = -vars2[i] * vars2[j] / var2_sum;
                variance_conditional.set(i, j, covariance);
                variance_conditional.set(j, i, covariance);
            }
        }

        // create the multivariate normal distribution
        NormalM norm = new NormalM(new DenseVector(mean_conditional), variance_conditional);

        // get samples
        List<Vec> samples_conditional = norm.sample(count, rng);

        // use the sample to generate the remaining component
        double[][] dependent_samples = new double[count][dof + 1];
        for (int i = 0; i < dof; i++) {
            double[] sample_i = samples_conditional.get(i).arrayCopy();
            System.arraycopy(sample_i, 0, dependent_samples[i], 0, dof);
            dependent_samples[i][dof] = z - DoubleStream.of(sample_i).sum();
        }

        return dependent_samples;
    }

    private double[][] mergeSamples(double[][] dependent_samples, double[][] independent_samples) {
        assert dependent_samples.length == independent_samples.length;

        // if there's only 1 distribution, then there's zero degrees of freedom
        // return the input value
        int N = dependent_samples.length;
        int num_vars = weights.length;

        double[][] samples = new double[N][num_vars];

        for (int i = 0; i < dependentIndices.length; i++) {
            final int idx = dependentIndices[i];
            for (int n = 0; n < N; n++) {
                samples[n][idx] = dependent_samples[n][i];
            }
        }

        for (int i = 0; i < independentIndices.length; i++) {
            final int idx = independentIndices[i];
            for (int n = 0; n < N; n++) {
                samples[n][idx] = independent_samples[n][i];
            }
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
            FilterDistribution activationDistribution, ActivationFunction activator, double weight) {
        return x -> activationDistribution.getProbabilityDensity(activator.inverse(x) / weight)
                * Math.abs(activator.inverseDerivative(x) / weight);
    }

}
