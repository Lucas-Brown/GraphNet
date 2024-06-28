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

import jsat.linear.Vec;

public class Trainer {

    private WeightsLinearizer linearizer;
    private NetworkInputEvaluater networkEvaluater;
    private ISolver weightsSolver;
    private ISolver probabilitySolver;

    private Vec weightsDeltas;
    private Vec probabilityDeltas;

    protected Double[][] inputs;
    protected Double[][] targets;

    protected WeightedAverage total_error;

    public Trainer(GraphNetwork network, ISolver weightsSolver) {
        this.weightsSolver = weightsSolver;

        linearizer = new WeightsLinearizer(network);
        networkEvaluater = new NetworkInputEvaluater(network);

        total_error = new WeightedAverage();
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
        History<Outcome, INode> history = networkEvaluater.computeNetworkInference();
        weightsDeltas = weightsSolver.solve(history);
        probabilityDeltas = probabilitySolver.solve(history);
        
        applyWeightDeltas();
        applyProbabilityDeltas();
    }

    private void applyWeightDeltas() {
        linearizer.allNodes.forEach(this::applyErrorSignalsToNode);
    }

    protected void applyErrorSignalsToNode(ITrainable node) {
        double[] allDeltas = weightsDeltas.arrayCopy();
        double[] gradient = linearizer.nodeSlice(node, allDeltas);
        node.applyDelta(gradient);
    }

    

    private void applyProbabilityDeltas() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'applyProbabilityDeltas'");
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

            double rate = Math.min(pass_rate/outcome.sourceTransferProbabilities[i], 1);
            outcome.sourceOutcomes[i].passRate.add(rate, outcome.probability);
        }

    }
}
