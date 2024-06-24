package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;

public class BackpropTrainer extends Trainer{

    public double epsilon = 0.01;
    private boolean normalizeError;

    public BackpropTrainer(GraphNetwork network, ErrorFunction errorFunction, boolean normalizeError) {
        super(network, errorFunction);
        this.normalizeError = normalizeError;
    }

    @Override
    protected void computeErrorOfNetwork(boolean print_forward) {
        computeErrorOfOutputs(print_forward);
        backpropagateErrors();
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

            double error = errorFunction.error(outcome.activatedValue, target);
            // assert Double.isFinite(errorFunction.error(outcome.activatedValue, target));
            total_error.add(error, outcome.probability);
        }

    }

    private void backpropagateErrors() {
        ArrayList<INode> nodes = network.getNodes();
        while (timestep > 0) {
            HashMap<INode, ArrayList<Outcome>> state = networkHistory.getStateAtTimestep(timestep);
            for (Entry<INode, ArrayList<Outcome>> e : state.entrySet()) {
                INode node = nodes.get(e.getKey().getID());
                updateNodeForTimestep((ITrainable) node, e.getValue());
            }
            timestep--;
        }
    }

    private void updateNodeForTimestep(ITrainable node, ArrayList<Outcome> outcomesAtTIme) {
        prepareOutputDistributionAdjustments(node, outcomesAtTIme);
        outcomesAtTIme.forEach(outcome -> sendErrorsBackwards(node, outcome));
        outcomesAtTIme.forEach(outcome -> adjustProbabilitiesForOutcome(node, outcome));
    }


    @Override
    protected void applyErrorSignals() {
        allNodes.forEach(this::applyErrorSignalsToNode);
    }

    private void applyErrorSignalsToNode(ITrainable node) {
        applyErrorSignals(node, networkHistory.getHistoryOfNode(node));
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
        if (node instanceof IInputNode || !outcomeAtTime.errorDerivative.hasValues()) {
            return;
        }

        int binary_string = outcomeAtTime.binary_string;
        double error_derivative = outcomeAtTime.errorDerivative.getAverage()
                * node.getActivationFunction().derivative(outcomeAtTime.netValue);

        double[] weightsOfNodes = node.getWeights(binary_string);

        for (int i = 0; i < weightsOfNodes.length; i++) {
            if (!outcomeAtTime.passRate.hasValues() || outcomeAtTime.probability == 0) {
                continue;
            }
            Outcome so = outcomeAtTime.sourceOutcomes[i];

            // Get the arc connectting the node to its source
            Arc arc = node.getIncomingConnectionFrom(outcomeAtTime.sourceNodes[i]).get();

            // use the arc to predict the probability of this event
            double shifted_value = so.activatedValue - error_derivative;
            double prob = outcomeAtTime.probability * arc.filter.getChanceToSend(shifted_value)
                    / outcomeAtTime.sourceTransferProbabilities[i];

            // accumulate error
            so.errorDerivative.add(error_derivative * weightsOfNodes[i], prob);

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

        // no source nodes to adjust
        if(node instanceof IInputNode){
            return;
        }

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

    public void applyErrorSignals(ITrainable node, List<ArrayList<Outcome>> allOutcomes) {
        double[] gradient = computeGradient(node, allOutcomes);
        for (int i = 0; i < gradient.length; i++) {
            gradient[i] *= epsilon;
        }
        node.applyDelta(gradient);
    }

    private double[] computeGradient(ITrainable node, List<ArrayList<Outcome>> allOutcomes) {
        if(node instanceof IInputNode){
            return new double[0];
        }

        double[] gradient = new double[node.getNumberOfVariables()];
        int T = 0;

        // for all time steps
        for (ArrayList<Outcome> outcomesAtTime : allOutcomes) {

            // Compute the probability volume of this timestep
            double probabilityVolume = 0;
            boolean atLeastOnePass = false;
            for (Outcome outcome : outcomesAtTime) {
                probabilityVolume += outcome.probability;
                atLeastOnePass |= outcome.passRate.hasValues();
            }

            // at least one outcome must have a chance to pass through
            if (!atLeastOnePass) {
                continue;
            }

            // Increase the number of non-zero timesteps
            T++;

            // if zero volume, move on to next set
            if (probabilityVolume == 0) {
                continue;
            }

            // add error to the gradient
            for (Outcome outcome : outcomesAtTime) {
                if (!outcome.passRate.hasValues() || outcome.errorDerivative.getProdSum() == 0) {
                    continue;
                }

                int key = outcome.binary_string;

                double error = outcome.errorDerivative.getProdSum();
                if(normalizeError){
                    error *= outcome.probability / probabilityVolume;
                }
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
