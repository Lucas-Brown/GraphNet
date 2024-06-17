package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.WeightedAverage;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;

import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;

public class NewtonTrainer {

    public double epsilon = 0.01;
    private WeightedAverage total_error;

    private int timestep;
    private final GraphNetwork network;
    private final History networkHistory;
    private final ErrorFunction errorFunction;

    private Double[][] inputs;
    private Double[][] targets;

    private ArrayList<OutputNode> outputNodes;
    private HashSet<ITrainable> allNodes;

    private boolean normalizeError;

    private int totalNumOfVariables;
    private HashMap<ITrainable, Integer> vectorNodeOffset;

    private Matrix errorDerivative;
    private Matrix errorHessian;

    public NewtonTrainer(GraphNetwork network, ErrorFunction errorFunction, boolean normalizeError) {
        this.network = network;
        this.errorFunction = errorFunction;
        this.normalizeError = normalizeError;
        networkHistory = new History(network);

        castAllToTrainable();

        network.setInputOperation(this::applyInputToNode);
        outputNodes = network.getOutputNodes();

        total_error = new WeightedAverage();
    }

    private void castAllToTrainable() {
        ArrayList<INode> nodes = network.getNodes();
        vectorNodeOffset = new HashMap<>(nodes.size());
        allNodes = new HashSet<>(nodes.size());

        int totalNumOfVariables = 0;
        for (INode node : nodes) {
            ITrainable tnode = (ITrainable) node;
            allNodes.add(tnode);
            vectorNodeOffset.put(tnode, totalNumOfVariables);
            totalNumOfVariables += tnode.getNumberOfVariables();
        }
    }

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public void trainNetwork(int steps, int print_interval) {
        while (steps-- > 0) {
            trainingStep(steps % print_interval == 0);
        }
    }

    public void trainingStep(boolean print_forward) {
        captureForward(print_forward);

        computeErrorOfOutputs(print_forward);
        backpropagateErrors();
        if(normalizeError){
            normalizeErrors();
        }
        applyErrorSignals();
        network.deactivateAll();
        networkHistory.burnHistory();
    }

    private int getLinearIndexOfWeight(ITrainable node, int key, int weight_index){
        return vectorNodeOffset.get(node) + node.getLinearIndexOfWeight(key, weight_index);
    }

    private int getLinearIndexOfBias(ITrainable node, int key){
        return vectorNodeOffset.get(node) + node.getLinearIndexOfBias(key);
    }

    private void captureForward(boolean print_forward) {
        for (timestep = 0; timestep < inputs.length; timestep++) {
            network.trainingStep();
            if (print_forward) {
                System.out.println(network.toString() + " | Target = " + Arrays.toString(targets[timestep]));
            }
            networkHistory.captureState();
        }
        timestep--;
    }

    private void computeErrorOfOutputs(boolean print_forward) {
        for (int time = timestep; time > 0; time--) {
            for (int i = 0; i < outputNodes.size(); i++) {
                computeErrorOfOutput(outputNodes.get(i), time, targets[time][i]);
            }
        }
        // assert total_error.getAverage() < 1E6;
        assert Double.isFinite(total_error.getAverage());
        if (print_forward) {
            System.out.println(total_error.getAverage());
        }
        total_error.reset();
    }

    private void computeErrorOfOutput(OutputNode node, int timestep, Double target) {
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node);
        if (outcomes == null) {
            return;
        }

        if (target == null) {
            for (Outcome outcome : outcomes) {
                outcome.passRate.add(0, 1);
            }
            return;
        }

        for (Outcome outcome : outcomes) {
            // if (timestep == this.timestep) {
            //     outcome.passRate.add(1 - 1d / timestep, 1);
            // } else {
                outcome.passRate.add(1, 1);
            // }
            outcome.errorDerivative.add(errorFunction.error_derivative(outcome.activatedValue, target),
                    outcome.probability);

            outcome.errorSecondDerivative += errorFunction.error_second_derivative(outcome.activatedValue, target) * outcome.probability;

            outcome.crossErrorDerivative = new DenseMatrix(totalNumOfVariables, 1); 

            double error = errorFunction.error(outcome.activatedValue, target);
            // assert Double.isFinite(errorFunction.error(outcome.activatedValue, target));
            total_error.add(error, outcome.probability);
        }

    }

    private void backpropagateErrors() {
        ArrayList<INode> nodes = network.getNodes();
        while (timestep >= 0) {
            HashMap<INode, ArrayList<Outcome>> state = networkHistory.getStateAtTimestep(timestep);
            for (Entry<INode, ArrayList<Outcome>> e : state.entrySet()) {
                INode node = nodes.get(e.getKey().getID());
                updateNodeForTimestep((ITrainable) node, e.getValue());
            }
            timestep--;
        }
    }

    
    private void normalizeErrors() {
        networkHistory.getAnonymousHistoryStream().forEach(this::normalizeErrors);
    }

    private void normalizeErrors(ArrayList<Outcome> outcomesAtTime){
        
        // compute the probability volume
        double probabilityVolume = 0;
        for (Outcome outcome : outcomesAtTime) {
            probabilityVolume += outcome.probability;
        }

        // if the volume is zero, we can't noramlize. This should only occure when the error is zero anyways
        if(probabilityVolume == 0){
            return;
        }
        for (Outcome outcome : outcomesAtTime) {
            double normalizationConstant = outcome.probability / probabilityVolume;
            double error_average = outcome.errorDerivative.getAverage();
            outcome.errorDerivative.reset();
            outcome.errorDerivative.add(error_average * normalizationConstant, 1);
            outcome.errorSecondDerivative *= normalizationConstant;
        }

    }


    private void updateNodeForTimestep(ITrainable node, ArrayList<Outcome> outcomesAtTIme) {
        prepareOutputDistributionAdjustments(node, outcomesAtTIme);
        outcomesAtTIme.forEach(outcome -> sendErrorsBackwards(node, outcome));
        outcomesAtTIme.forEach(outcome -> adjustProbabilitiesForOutcome(node, outcome));
    }

    private void applyErrorSignals() {
        allNodes.forEach(this::applyErrorSignalsToNode);
        allNodes.forEach(ITrainable::applyDistributionUpdate);
        allNodes.forEach(ITrainable::applyFilterUpdate);
    }

    private void applyErrorSignalsToNode(ITrainable node) {
        applyErrorSignals(node, networkHistory.getHistoryOfNode(node));
    }

    private void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap) {
        applyInputToNode(inputNodeMap, inputs, timestep);
    }

    /**
     * Use the outcomes to prepare weighted adjustments to the outcome distribution
     */
    public void prepareOutputDistributionAdjustments(ITrainable node, ArrayList<Outcome> allOutcomes) {
        IExpectationAdjuster adjuster = node.getOutputDistributionAdjuster();
        for (Outcome o : allOutcomes) {
            // weigh the outcomes by their probability of occurring
            // double error = o.errorOfOutcome.hasValues() ? o.errorOfOutcome.getAverage() :
            // 0;
            double error = 0;
            adjuster.prepareAdjustment(o.probability, new double[] { o.activatedValue - error });
        }
    }

    public void sendErrorsBackwards(ITrainable node, Outcome outcomeAtTime) {
        if (!outcomeAtTime.errorDerivative.hasValues()) {
            return;
        }

        int binary_string = outcomeAtTime.binary_string;
        double activator_derivative = node.getActivationFunction().derivative(outcomeAtTime.netValue);
        double activator_second_derivative = node.getActivationFunction().secondDerivative(outcomeAtTime.netValue);
        double error_derivative = outcomeAtTime.errorDerivative.getAverage();
        double net_derivative = error_derivative * activator_derivative;

        double[] weightsOfNodes = node.getWeights(binary_string);

        for (int i = 0; i < weightsOfNodes.length; i++) {
            if (!outcomeAtTime.passRate.hasValues() || outcomeAtTime.probability == 0) {
                continue;
            }
            double w = weightsOfNodes[i];
            Outcome so = outcomeAtTime.sourceOutcomes[i];

            // Get the arc connectting the node to its source
            Arc arc = node.getIncomingConnectionFrom(outcomeAtTime.sourceNodes[i]).get();

            // use the arc to predict the probability of this event
            double shifted_value = so.activatedValue - error_derivative;
            double prob = outcomeAtTime.probability * arc.filter.getChanceToSend(shifted_value)
                    / outcomeAtTime.sourceTransferProbabilities[i];

            // accumulate first derivative error
            so.errorDerivative.add(net_derivative * w, prob);

            
            // accumulate second derivative error
            double secondDerivative = so.errorSecondDerivative * net_derivative * net_derivative;
            secondDerivative += error_derivative * activator_second_derivative * w * w;
            so.errorSecondDerivative += secondDerivative * prob;

            // compute the cross-derivative
            so.crossErrorDerivative = so.crossErrorDerivative.add(errorDerivative);

            // accumulate pass rates
            double pass_avg = outcomeAtTime.passRate.getAverage();
            assert Double.isFinite(pass_avg);
            so.passRate.add(pass_avg, prob);

            // apply error as new point for the distribution
            // Arc connection =
            // outcomeAtTime.sourceNodes[i].getOutgoingConnectionTo(this).get();
            // connection.probDist.prepareReinforcement(outcomeAtTime.netValue -
            // error_derivative);
        }

    }

    public void adjustProbabilitiesForOutcome(ITrainable node, Outcome outcome) {
        if (!outcome.passRate.hasValues() || outcome.probability == 0) {
            return;
        }
        double pass_rate = outcome.passRate.getAverage();

        // Add another point for the net firing chance distribution
        IExpectationAdjuster adjuster = node.getSignalChanceDistributionAdjuster();
        adjuster.prepareAdjustment(outcome.probability, new double[] { pass_rate });

        // Reinforce the filter with the pass rate for each point
        for (int i = 0; i < outcome.sourceNodes.length; i++) {
            INode sourceNode = outcome.sourceNodes[i];

            double error_derivative = outcome.errorDerivative.getAverage();
            Arc arc = node.getIncomingConnectionFrom(sourceNode).get(); // should be guaranteed to exist

            // if the error is not defined and the pass rate is 0, then zero error should be
            // expected
            if (Double.isNaN(error_derivative) && pass_rate == 0) {
                error_derivative = 0;
            }
            assert !Double.isNaN(error_derivative);

            if (arc.filterAdjuster != null) {
                double shifted_value = outcome.sourceOutcomes[i].activatedValue - error_derivative;
                double prob = outcome.probability * arc.filter.getChanceToSend(shifted_value)
                        / outcome.sourceTransferProbabilities[i];
                arc.filterAdjuster.prepareAdjustment(prob, new double[] { shifted_value, pass_rate });
            }
        }

    }

    public static void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap, Double[][] input,
            int counter) {
        InputNode[] sortedNodes = inputNodeMap.values().stream().sorted().toArray(InputNode[]::new);

        for (int i = 0; i < sortedNodes.length; i++) {
            if (input[counter][i] != null) {
                sortedNodes[i].acceptUserForwardSignal(input[counter][i]);
            }
        }
    }

    public void applyErrorSignals(ITrainable node, List<ArrayList<Outcome>> allOutcomes) {
        double[] gradient = computeGradient(node, allOutcomes);
        node.applyGradient(gradient, epsilon);
    }

    private double[] computeGradient(ITrainable node, List<ArrayList<Outcome>> allOutcomes) {

        double[] gradient = new double[node.getNumberOfVariables()];
        int T = 0;

        // for all time steps
        for (ArrayList<Outcome> outcomesAtTime : allOutcomes) {

            boolean atLeastOnePass = false;
            for (Outcome outcome : outcomesAtTime) {
                atLeastOnePass |= outcome.passRate.hasValues();
            }

            // at least one outcome must have a chance to pass through
            if (!atLeastOnePass) {
                continue;
            }

            // Increase the number of non-zero timesteps
            T++;


            // add error to the gradient
            for (Outcome outcome : outcomesAtTime) {
                if (!outcome.passRate.hasValues() || outcome.errorDerivative.getProdSum() == 0) {
                    continue;
                }

                int key = outcome.binary_string;

                double error = outcome.errorDerivative.getProdSum();
                assert Double.isFinite(error);
                gradient[node.getLinearIndexOfBias(key)] += error;

                for (int i = 0; i < node.getWeights(key).length; i++) {
                    gradient[node.getLinearIndexOfWeight(key, i)] += error * outcome.sourceOutcomes[i].netValue;
                }
            }
        }

        // divide all gradients by the number of non-empty timesteps
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] /= T;
        }

        return gradient;
    }

}
