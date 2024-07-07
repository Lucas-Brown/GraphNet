package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.NetworkTraining.OutputDerivatives.IGradient;

/**
 * This class is intentionally broken for now. 
 * Want to keep the core logic of the distribution perspective without the work of fully reorganizing that structure to accomodate the new method 
 */
public class DistributionAdjusterGradient implements IGradient {
 
    
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
            Edge arc = node.getIncomingConnectionFrom(sourceNode).get(); // should be guaranteed to exist

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
