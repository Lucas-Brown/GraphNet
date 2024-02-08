

import static org.junit.Assert.assertEquals;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;


/**
 * Test the convergence of randomized path decisions
 */
public class FixedChanceProofOfConcept {
    
    private static final double DELTA = 1E-2;
    private static final double STEP_SIZE = 1E-1;
    private static final double TOLLERANCE = 1E-12;

    private Random rng;
    private ArrayList<Double> initial_set;
    private ArrayList<Double> trial_set;

    private static final double fixed_chance_1 = 0.6;
    private static final double fixed_chance_2 = 0.8;

    private static final double w1 = fixed_chance_1*(1-fixed_chance_2);
    private static final double w2 = fixed_chance_2*(1-fixed_chance_1);
    private static final double w3 = fixed_chance_1*fixed_chance_2;
    private static final double net_weight = w1+w2+w3;

    private double[] initial_params;

    public FixedChanceProofOfConcept(int initial_data_size, int trial_size)
    {
        rng = new Random();
        initial_set = getRandomSet(initial_data_size);
        trial_set = getRandomSet(trial_size);

        double[] initial_guess = new double[]{-1, 0.5, 2, 2};
        //ToDoubleBiFunction<double[], Integer> mleFunction = (params, idx) -> logLikelihoodDerivativeFixedChance(params, initial_set, idx);
        initial_params = maximumLikelihoodEstimate(initial_guess, new logLikelihoodFunction(initial_set));
    }

    public double[] getExactParams()
    {
        ArrayList<Double> all_data = new ArrayList<>(initial_set);
        all_data.addAll(trial_set);

        //ToDoubleBiFunction<double[], Integer> mleFunction = (params, idx) -> logLikelihoodDerivativeFixedChance(params, all_data, idx);
        return maximumLikelihoodEstimate(initial_params, new logLikelihoodFunction(all_data));
    }


    public double[] getTrueMeanAndVariance()
    {
        ArrayList<Double> all_data = new ArrayList<>(initial_set);
        all_data.addAll(trial_set);

        double mean = all_data.stream().mapToDouble(x -> x).average().getAsDouble();
        double var2 = all_data.stream().mapToDouble(x -> (x - mean)*(x-mean)).average().getAsDouble();
        return new double[]{mean, Math.sqrt(var2)};
    }

    public double[][] getParamsFromRandomizedTrial(int trial_count, Trial trial)
    {
        Gaussian g1_initial = new Gaussian(initial_params[0], initial_params[1], initial_set.size());
        Gaussian g2_initial = new Gaussian(initial_params[2], initial_params[3], initial_set.size());

        double[][] param_outcomes = new double[trial_count][4];
        for(int trial_num = 0; trial_num < trial_count; trial_num++)
        {
            Gaussian g1 = new Gaussian(g1_initial);
            Gaussian g2 = new Gaussian(g2_initial);

            Collections.shuffle(trial_set);
            for(Double x : trial_set)
            {
                trial.applyTrial(g1, g2, x);
            }
            
            param_outcomes[trial_num][0] = g1.getMean();
            param_outcomes[trial_num][1] = g1.getVariance();
            param_outcomes[trial_num][2] = g2.getMean();
            param_outcomes[trial_num][3] = g2.getVariance();
        }

        return param_outcomes;
    }

    private void applyFixedChanceTrial(Gaussian g1, Gaussian g2, Double x)
    {
        switch (selectRandom(w1, w2, w3)) {
            case 0:
                g1.addPoint(x, w1);
                break;
            case 1:
                g2.addPoint(x, w2);
                break;
            case 2:
                //?? Maybe ??
                double mu1 = g1.getMean();
                g1.addPoint(x-g2.getMean(), w3/2);
                g2.addPoint(x-mu1, w3/2);
                break;
        
            default:
                break;
        }
    }

    
    private void applyVariableChanceTrial(Gaussian g1, Gaussian g2, Double x)
    {
        double chance_1 = g1.evaluateAt(x);
        double chance_2 = g2.evaluateAt(x);

        double w1 = chance_1*(1-chance_2);
        double w2 = chance_2*(1-chance_1);
        double w3 = chance_1*chance_2;

        switch (selectRandom(w1, w2, w3)) {
            case 0:
                g1.addPoint(x, w1);
                break;
            case 1:
                g2.addPoint(x, w2);
                break;
            case 2:
                //?? Maybe ??
                double mu1 = g1.getMean();
                g1.addPoint(x-g2.getMean(), w3/2);
                g2.addPoint(x-mu1, w3/2);
                break;
        
            default:
                break;
        }
    }

    private int selectRandom(double... weights)
    {
        double total = DoubleStream.of(weights).sum();
        double roll = rng.nextDouble() * total;

        double sum = 0;
        for(int i = 0; i < weights.length; i++)
        {
            sum += weights[i];
            if(sum > roll)
            {
                return i;
            }
        }

        return -1;
    }

    private ArrayList<Double> getRandomSet(int set_size) {
        return DoubleStream.generate(this::randomRoll).limit(set_size).boxed().collect(Collectors.toCollection(ArrayList::new));
    }

    private double randomRoll()
    {
        switch (selectRandom(w1, w2, w3)) {
            case 0:
                return rng.nextGaussian();
            case 1:
                return rng.nextGaussian() + 1;
            default:
                return rng.nextGaussian() + rng.nextGaussian() + 1;
        }
    }

    private double[] maximumLikelihoodEstimate(double[] guess, Function log_likelihood)
    {
        NelderMead nm = new NelderMead();
        List<Vec> init_points = new ArrayList<>(1);

        init_points.add(new DenseVector(guess)); 
        init_points.add(new DenseVector(new double[]{guess[0] + 0.5, guess[1], guess[2], guess[3]})); 
        init_points.add(new DenseVector(new double[]{guess[0], guess[1], guess[2] + 0.5, guess[3]})); 
        init_points.add(new DenseVector(new double[]{guess[0], guess[1]*1.2, guess[2], guess[3]})); 
        init_points.add(new DenseVector(new double[]{guess[0], guess[1], guess[2], guess[3]*1.2})); 

        Vec solution = nm.optimize(TOLLERANCE, 100000, log_likelihood, init_points);
        return solution.arrayCopy();
    }

    private double[] maximumLikelihoodEstimate(double[] guess, ToDoubleBiFunction<double[], Integer> log_likelihood_derivative)
    {

        double[] next_guess;
        boolean any_change;

        do
        {
            any_change = false; 
            next_guess = new double[guess.length];

            for(int i = 0; i < next_guess.length; i++)
            {
                final int j = i;
                ToDoubleFunction<double[]> log_likelihood_derivative_i = params -> log_likelihood_derivative.applyAsDouble(params, j);
                double log_likelihood_derivative_of_guess = log_likelihood_derivative_i.applyAsDouble(guess);
                if(log_likelihood_derivative_of_guess == 0)
                {
                    next_guess[j] = guess[j];
                    continue;
                }
                next_guess[j] = guess[j] - STEP_SIZE * log_likelihood_derivative_of_guess / estimateDerivative(guess, log_likelihood_derivative_i, j);
                any_change |= Math.abs(next_guess[j] - guess[j]) >= TOLLERANCE;
            }

            guess = next_guess;

        }while(any_change);

        return guess;
    }

    private double estimateDerivative(double[] params, ToDoubleFunction<double[]> func, int with_respect_to_index)
    {
        double[] params_lower = params.clone();
        double[] params_upper = params.clone();

        params_lower[with_respect_to_index] -= DELTA;
        params_upper[with_respect_to_index] += DELTA;

        double lower = func.applyAsDouble(params_lower);
        double upper = func.applyAsDouble(params_upper);

        return (upper - lower) / (2*DELTA);
    }

    private double logLikelihoodDerivativeFixedChance(double[] params, ArrayList<Double> data, int derivIndex)
    {
        ToDoubleFunction<Double> deriveFunc;
        switch (derivIndex) {
            case 0:
                deriveFunc = x -> logLikelihoodDerivative1FixedChancePoint(params, x);
                break;
            case 1:
                deriveFunc = x -> logLikelihoodDerivative2FixedChancePoint(params, x);
                break;
            case 2:
                deriveFunc = x -> logLikelihoodDerivative3FixedChancePoint(params, x);
                break;
            case 3:
                deriveFunc = x -> logLikelihoodDerivative4FixedChancePoint(params, x);
                break;
        
            default:
                return 0;
        }
        return data.stream().mapToDouble(deriveFunc).sum();
    }

    private double logLikelihoodDerivative1FixedChancePoint(double[] params, double x)
    {
        double var_combined = Math.sqrt(params[1]*params[1] + params[3]*params[3]);

        double divisor = 0;
        divisor += w1*Gaussian.Gauss(params[0], params[1], x);
        divisor += w2*Gaussian.Gauss(params[2], params[3], x);
        divisor += w3*Gaussian.Gauss(params[0]+params[2], var_combined, x);

        double numerator = 0;
        numerator += w1*Gaussian.GaussMeanDerivative(params[0], params[1], x);
        numerator += w3*Gaussian.GaussMeanDerivative(params[0]+params[2], var_combined, x);
        return numerator/divisor;
    }

    private double logLikelihoodDerivative2FixedChancePoint(double[] params, double x)
    {
        double var_combined = Math.sqrt(params[1]*params[1] + params[3]*params[3]);

        double divisor = 0;
        divisor += w1*Gaussian.Gauss(params[0], params[1], x);
        divisor += w2*Gaussian.Gauss(params[2], params[3], x);
        divisor += w3*Gaussian.Gauss(params[0]+params[2], var_combined, x);

        double numerator = 0;
        numerator += w1*Gaussian.GaussVarianceDerivative(params[0], params[1], x);
        numerator += w3*Gaussian.GaussVarianceDerivative(params[0]+params[2], var_combined, x)* var_combined/params[1];
        return numerator/divisor;
    }

    
    private double logLikelihoodDerivative3FixedChancePoint(double[] params, double x)
    {
        double var_combined = Math.sqrt(params[1]*params[1] + params[3]*params[3]);

        double divisor = 0;
        divisor += w1*Gaussian.Gauss(params[0], params[1], x);
        divisor += w2*Gaussian.Gauss(params[2], params[3], x);
        divisor += w3*Gaussian.Gauss(params[0]+params[2], var_combined, x);

        double numerator = 0;
        numerator += w2*Gaussian.GaussMeanDerivative(params[2], params[3], x);
        numerator += w3*Gaussian.GaussMeanDerivative(params[0]+params[2], var_combined, x);
        return numerator/divisor;
    }

    private double logLikelihoodDerivative4FixedChancePoint(double[] params, double x)
    {
        double var_combined = Math.sqrt(params[1]*params[1] + params[3]*params[3]);

        double divisor = 0;
        divisor += w1*Gaussian.Gauss(params[0], params[1], x);
        divisor += w2*Gaussian.Gauss(params[2], params[3], x);
        divisor += w3*Gaussian.Gauss(params[0]+params[2], var_combined, x);

        double numerator = 0;
        numerator += w2*Gaussian.GaussVarianceDerivative(params[2], params[3], x);
        numerator += w3*Gaussian.GaussVarianceDerivative(params[0]+params[2], var_combined, x)* var_combined/params[3];
        return numerator/divisor;
    }

    public void testIterativeConvergence()
    {
        Random rng = new Random();
        int data_size = 100;
        List<Double> dataset = DoubleStream.generate(rng::nextGaussian).limit(data_size).boxed().collect(Collectors.toList());
        
        double mean = dataset.stream().mapToDouble(x -> x).average().getAsDouble();
        double var = dataset.stream().mapToDouble(x -> (x - mean)*(x-mean)).average().getAsDouble();
        var = Math.sqrt(var);

        int sample_size = 10;
        ArrayList<Double> sample = new ArrayList<Double>(sample_size);
        for(int i = 0; i < sample_size; i++)
        {
            sample.add(dataset.remove(0));
        }

        double mean_initial = sample.stream().mapToDouble(x -> x).average().getAsDouble();
        double var_initial = sample.stream().mapToDouble(x -> (x - mean)*(x-mean)).average().getAsDouble();
        var_initial = Math.sqrt(var_initial);

        Gaussian gaussian = new Gaussian(mean_initial, var_initial, sample_size);
        dataset.forEach(x -> gaussian.addPoint(x, 1));

        assertEquals(gaussian.mean, mean, 1E-6);
        assertEquals(gaussian.variance, var, 1E-6);
    }

    public static void main(String[] args)
    {
        FixedChanceProofOfConcept fcpc = new FixedChanceProofOfConcept(1000, 1000);
        double[] true_params = fcpc.getTrueMeanAndVariance();
        double[] exact_params = fcpc.getExactParams();
        double[][] trial_params = fcpc.getParamsFromRandomizedTrial(10, fcpc::applyVariableChanceTrial);

        System.out.println("True mean: " + true_params[0] + ", true variance: " + true_params[1]);
        System.out.println("[" + (exact_params[0] + exact_params[2]) + ", " + Math.sqrt(exact_params[1]*exact_params[1] + exact_params[3]*exact_params[3]) + "]");
        System.out.println("{");
        for(double[] arr : trial_params)
        {
            System.out.println("[" + (arr[0] + arr[2]) + ", " + Math.sqrt(arr[1]*arr[1] + arr[3]*arr[3]) + "]");
        }
        System.out.println("}");
        System.out.println();
    }

    public static class Gaussian
    {
        private double mean;
        private double variance;
        private double N;

        public Gaussian(double mean, double varicance, double N)
        {
            this.mean = mean;
            this.variance = varicance;
            this.N = N;
        }

        public Gaussian(Gaussian g)
        {
            mean = g.mean;
            variance = g.variance;
            N = g.N;
        }

        public double evaluateAt(double x)
        {
            return Gauss(mean, variance, x);
        }

        public void addPoint(double x, double weight)
        {
            double N_new = N + weight;
            double d = x - mean;
            mean += weight/N_new * d;
            variance = Math.sqrt((N) / (N_new) * (variance*variance + weight/(N_new) * d*d));
            N = N_new;
        }

        private double getMean() {
            return mean;
        }

        private double getVariance() {
            return variance;
        }

        public static double Gauss(double mean, double variance, double x)
        {
            double var2 = variance*variance;
            double d = x - mean;
            return Math.exp(-d*d/(2 * var2)) / Math.sqrt(2*Math.PI*var2);
        }

        public static double GaussMeanDerivative(double mean, double variance, double x)
        {
            double var2 = variance*variance;
            double d = x - mean;
            return d * Math.exp(-d*d/(2 * var2)) / (var2 * Math.sqrt(2*Math.PI*var2));
        }

        public static double GaussVarianceDerivative(double mean, double variance, double x)
        {
            double var2 = variance*variance;
            double d = x - mean;
            return (d*d/(var2*variance) - 1/variance) * Math.exp(-d*d/(2 * var2)) / Math.sqrt(2*Math.PI*var2);
        }
        
    }

    private static class logLikelihoodFunction implements Function
    {
        private ArrayList<Double> data;

        public logLikelihoodFunction(ArrayList<Double> data)
        {
            this.data = data;
        }
        
        private double logLikelihoodFixedChancePoint(double[] params, double x)
        {
            double sum = 0;
            sum += fixed_chance_1*(1-fixed_chance_2)*Gaussian.Gauss(params[0], params[1], x);
            sum += fixed_chance_2*(1-fixed_chance_1)*Gaussian.Gauss(params[2], params[3], x);
            sum += fixed_chance_1*fixed_chance_2*Gaussian.Gauss(params[0]+params[2], Math.sqrt(params[1]*params[1] + params[3]*params[3]), x);
            sum = Math.log(sum);
            sum -= Math.log(net_weight);
            return sum;
        }

        @Override
        public double f(final double... params) {
            return -data.stream().mapToDouble(x -> logLikelihoodFixedChancePoint(params, x)).sum(); // negative so that the minimizer attempts to maximize instead
        }

        @Override
        public double f(Vec params) {
            return f(params.arrayCopy());
        }
    }

    @FunctionalInterface
    interface Trial
    {
        public abstract void applyTrial(Gaussian g1, Gaussian g2, double x);
    }


}
