package com.lucasbrown.NetworkTraining;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Supplier;
import java.util.stream.IntStream;

import org.junit.Assert;
import org.junit.Test;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Global.NodeBuilder;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Filters.FlatRateFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.NormalPeakFilter;
import com.lucasbrown.GraphNetwork.Local.Filters.OpenFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.SimpleProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ComplexCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.SimpleCombinator;
import com.lucasbrown.NetworkTraining.History.History;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.ForwardNetworkGradient;
import com.lucasbrown.NetworkTraining.Solvers.ADAMSolver;
import com.lucasbrown.NetworkTraining.Trainers.Trainer;
import com.lucasbrown.NetworkTraining.Trainers.WeightsLinearizer;

import jsat.linear.Vec;

public class ExponentialTest {

    private static double base = 2;
    private int N = 50;
    private Double[][][] inputData;
    private Double[][][] outputData;

    private void initializeInputData() {
        inputData = new Double[][][]{ createInput(1), createInput(-2), createInput(0.3) };
    }

    private Double[][] createInput(double init_value){
        Double[][] input = new Double[N][1];
        input[0] = new Double[] { init_value };
        for (int i = 1; i < N; i++) {
            input[i] = new Double[] { null };
        }
        return input;
    }

    private void initializeOutputData() {
        outputData = new Double[inputData.length][][];
        for(int i = 0; i < inputData.length; i++){
            outputData[i] = createOutput(inputData[i]);
        }
    }

    private Double[][] createOutput(Double[][] input){
        Double[][] out = new Double[N][1];
        out[0] = new Double[] { null };
        out[1] = new Double[] { null };
        out[2] = new Double[] { input[0][0] };

        for (int i = 3; i < N; i++) {
            out[i] = new Double[] { out[i - 1][0] * base };
        }
        return out;
    }

    public GraphNetwork initializeNetwork(){
        GraphNetwork net = new GraphNetwork();

        NodeBuilder nodeBuilder = new NodeBuilder(net);

        Supplier<IFilter> filterSupplier = () -> new FlatRateFilter(0.999);
        nodeBuilder.setActivationFunction(ActivationFunction.LINEAR);
        nodeBuilder.setValueCombinator(ComplexCombinator::new);
        nodeBuilder.setProbabilityCombinator(() -> new SimpleProbabilityCombinator(filterSupplier));
        
        nodeBuilder.setAsInputNode();
        InputNode in = (InputNode) nodeBuilder.build();

        nodeBuilder.setAsHiddenNode();
        INode hidden = nodeBuilder.build();

        nodeBuilder.setAsOutputNode();
        OutputNode out = (OutputNode) nodeBuilder.build();

        in.setName("Input");
        hidden.setName("Hidden");
        out.setName("Output");

        net.addNewConnection(in, hidden);
        net.addNewConnection(hidden, hidden);
        net.addNewConnection(hidden, out);
        return net;
    }


    @Test
    /**
     * I *could* properly seperate out the tests, but I don't want to  
     */
    public void derivativeAssesment(){
        ExponentialTest exponentialGrowth = new ExponentialTest();
        exponentialGrowth.initializeInputData();
        exponentialGrowth.initializeOutputData();

        GraphNetwork net = exponentialGrowth.initializeNetwork();
        Trainer trainer = Trainer.getDefaultTrainer(net);
        trainer.setTrainingData(exponentialGrowth.inputData, exponentialGrowth.outputData);

        InputNode in = null;
        INode hidden = null;
        OutputNode out = null;
        for(INode node : net.getNodes()){
            if(node instanceof OutputNode){
                out = (OutputNode) node;
            }
            else if(node instanceof InputNode){
                in = (InputNode) node;
            }
            else{
                hidden = node;
            }
        }


        trainer.networkEvaluater.setInputData(exponentialGrowth.inputData[0]);
        NetworkHistory history = trainer.networkEvaluater.computeNetworkInference();

        // test to make sure the history aligns with the analytic form
        historyMatchesExpected(net, history, in, hidden, out);

        ForwardNetworkGradient forwardGradient = new ForwardNetworkGradient(trainer.weightLinearizer);
        ArrayList<HashMap<Outcome, Vec>> gradient = forwardGradient.getGradient(history);

        // test to make sure 
        gradientMatchesExpected(net, history, gradient, trainer.weightLinearizer, hidden);
    }

    private double getHiddenState(int timestep, INode hidden, double initial){
        int n = timestep - 1;
        IValueCombinator comb = hidden.getValueCombinator();
        double w = comb.getWeights(0b10)[0];
        double b = comb.getBias(0b10);

        double value = initial*Math.pow(w, n);
        value += b*IntStream.range(0, n).mapToDouble(t -> Math.pow(w, t)).sum(); 
        return value;
    }

    
    private double[] getHiddenDerivative(int timestep, INode hidden, double initial){
        int n = timestep - 1;
        IValueCombinator comb = hidden.getValueCombinator();
        double w = comb.getWeights(0b10)[0];
        double b = comb.getBias(0b10);

        double d_w;
        if(n == 0){
            d_w = 0;
        }
        else{
            d_w = n*initial*Math.pow(w, n-1);
            d_w += b*IntStream.range(1, n).mapToDouble(t -> t*Math.pow(w, t-1)).sum();
        }

        double d_b = IntStream.range(0, n).mapToDouble(t -> Math.pow(w, t)).sum();
        return new double[]{d_w, d_b};
    }

    
    private double getInitialValue(NetworkHistory history, INode hidden) {
        HashMap<INode, ArrayList<Outcome>> secondRecord = history.getStateAtTimestep(1);
        ArrayList<Outcome> secondOutcomes = secondRecord.get(hidden);

        double initialExponentialValue = secondOutcomes.get(0).activatedValue;
        return initialExponentialValue;
    }

    private void historyMatchesExpected(GraphNetwork net, NetworkHistory history, InputNode in, INode hidden, OutputNode out) {
        HashMap<INode, ArrayList<Outcome>> firstRecord = history.getStateAtTimestep(0);
        ArrayList<Outcome> firstOutcomes = firstRecord.get(in);
        Assert.assertNotNull(firstOutcomes);

        double initialExponentialValue = getInitialValue(history, hidden);

        for (int timestep = 2; timestep < history.getNumberOfTimesteps(); timestep++) {
            HashMap<INode, ArrayList<Outcome>> record = history.getStateAtTimestep(timestep);
            ArrayList<Outcome> outcomes = record.get(hidden);
            double hiddenValue = outcomes.get(0).activatedValue;
            double expectedHiddenValue = getHiddenState(timestep, hidden, initialExponentialValue);
            assertEquals(expectedHiddenValue, hiddenValue, 1E-12);
            // don't really care about the output node
        }
    }


    private void gradientMatchesExpected(GraphNetwork net, NetworkHistory history, ArrayList<HashMap<Outcome, Vec>> gradient, WeightsLinearizer linearizer, INode hidden) {
        double initialExponentialValue = getInitialValue(history, hidden);

        for (int timestep = 1; timestep < history.getNumberOfTimesteps(); timestep++) {
            
            HashMap<INode, ArrayList<Outcome>> record = history.getStateAtTimestep(timestep);
            Outcome outcome = record.get(hidden).get(0);

            Vec fullGradient = gradient.get(timestep).get(outcome);
            int weight_idx = linearizer.getLinearIndexOfWeight(hidden, 0b10, 0);
            int bias_idx = linearizer.getLinearIndexOfBias(hidden, 0b10);

            double[] hiddenGradient = {fullGradient.get(weight_idx), fullGradient.get(bias_idx)};
            double[] expectedHiddenGradient = getHiddenDerivative(timestep, hidden, initialExponentialValue);
            assertArrayEquals(expectedHiddenGradient, hiddenGradient, 1E-12);
            // don't really care about the output node
        }

        
    }

    public static void main(String[] args) {
        ExponentialTest exponentialGrowth = new ExponentialTest();
        exponentialGrowth.initializeInputData();
        exponentialGrowth.initializeOutputData();

        GraphNetwork net = exponentialGrowth.initializeNetwork();
        Trainer trainer = Trainer.getDefaultTrainer(net);
        trainer.setTrainingData(exponentialGrowth.inputData, exponentialGrowth.outputData);
        // ADAMSolver weightSolver = (ADAMSolver) trainer.weightsSolver;
        // weightSolver.alpha = 1E-4;
        // weightSolver.epsilon = 1E-6;   
        // weightSolver.beta_1 = 0.99;
        // weightSolver.beta_2 = 0.999;

        ADAMSolver probabilitySolver = (ADAMSolver) trainer.probabilitySolver;
        probabilitySolver.alpha = 0.1;
        probabilitySolver.epsilon = 1E-10;
        probabilitySolver.beta_1 = 0.99;
        probabilitySolver.beta_2 = 0.999;

        trainer.trainNetwork(10000000, 1000);

        // weightSolver.alpha = 1E-12;
        // weightSolver.epsilon = 1E-20;
        // weightSolver.beta_1 = 0.9;
        // weightSolver.beta_2 = 0.99;

        // trainer.trainNetwork(1000000, 1000);
        
        // weightSolver.alpha = 1E-12;
        // weightSolver.epsilon = 1E-20;

        // trainer.trainNetwork(1000000, 1000);


        net.deactivateAll();
       
    }
}
