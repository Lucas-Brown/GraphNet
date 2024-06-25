package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
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

public abstract class Trainer {

    protected int timestep;
    protected final GraphNetwork network;
    protected final History networkHistory;
    protected final ErrorFunction errorFunction;

    protected Double[][] inputs;
    protected Double[][] targets;

    protected ArrayList<OutputNode> outputNodes;
    protected HashSet<ITrainable> allNodes;

    protected WeightedAverage total_error;

    public Trainer(GraphNetwork network, ErrorFunction errorFunction) {
        this.network = network;
        this.errorFunction = errorFunction;
        networkHistory = new History(network);

        castAllToTrainable();

        network.setInputOperation(this::applyInputToNode);
        outputNodes = network.getOutputNodes();
        total_error = new WeightedAverage();
    }

    private void castAllToTrainable() {
        ArrayList<INode> nodes = network.getNodes();
        allNodes = new HashSet<>(nodes.size());
        for (INode node : nodes) {
            allNodes.add((ITrainable) node);
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

        computeErrorOfNetwork(print_forward);
        updateDistributions();
        applyErrorSignals();
        applyDistributionUpdate();
        network.deactivateAll();
        networkHistory.burnHistory();
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

    private void updateDistributions(){
        ArrayList<INode> nodes = network.getNodes();
        while (timestep > 0) {
            HashMap<INode, ArrayList<Outcome>> state = networkHistory.getStateAtTimestep(timestep);
            for (Entry<INode, ArrayList<Outcome>> e : state.entrySet()) {
                INode node = nodes.get(e.getKey().getID());
                updateDistributionsForTimestep((ITrainable) node, e.getValue());
            }
            timestep--;
        }
    }

    private void updateDistributionsForTimestep(ITrainable node, ArrayList<Outcome> outcomesAtTIme) {
        prepareOutputDistributionAdjustments(node, outcomesAtTIme);
        outcomesAtTIme.forEach(outcome -> adjustProbabilitiesForOutcome(node, outcome));
    }


    private void applyDistributionUpdate(){
        allNodes.forEach(ITrainable::applyDistributionUpdate);
        allNodes.forEach(ITrainable::applyFilterUpdate);
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
            adjuster.prepareAdjustment(o.probability, new double[] { o.activatedValue });
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

        // no source nodes to adjust
        if(node instanceof IInputNode){
            return;
        }

        // Reinforce the filter with the pass rate for each point
        for (int i = 0; i < outcome.sourceNodes.length; i++) {
            INode sourceNode = outcome.sourceNodes[i];
            Arc arc = node.getIncomingConnectionFrom(sourceNode).get(); // should be guaranteed to exist

            if (arc.filterAdjuster != null) {
                double activated_value = outcome.sourceOutcomes[i].activatedValue;
                double prob = outcome.probability * arc.filter.getChanceToSend(activated_value)
                        / outcome.sourceTransferProbabilities[i];
                arc.filterAdjuster.prepareAdjustment(prob, new double[] { activated_value, pass_rate });
            }

            outcome.sourceOutcomes[i].passRate.add(pass_rate, outcome.probability);
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

    protected abstract void computeErrorOfNetwork(boolean print_forward);

    protected abstract void applyErrorSignals() ;
}
