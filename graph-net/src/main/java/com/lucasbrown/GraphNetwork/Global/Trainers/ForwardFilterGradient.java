package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class ForwardFilterGradient implements INetworkGradient{

    protected FilterLinearizer linearizer;
    protected History<Outcome, INode> networkHistory;

    private ArrayList<HashMap<Outcome, Vec>> gradientsThroughTime;

    public ForwardFilterGradient(FilterLinearizer linearizer) {
        this.linearizer = linearizer;
    }

    @Override
    public ArrayList<HashMap<Outcome, Vec>> getGradient(History<Outcome, INode> networkHistory) {
        this.networkHistory = networkHistory;
        int n_steps = networkHistory.getNumberOfTimesteps();
        gradientsThroughTime = new ArrayList<>(n_steps);

        for(int timestep = 0; timestep < n_steps; timestep++){
            gradientsThroughTime.add(getGradientAtTime(timestep));
        }

        return gradientsThroughTime;
    }
    
    
    private HashMap<Outcome, Vec> getGradientAtTime(int timestep) {
        HashMap<Outcome, Vec> gradientMap = new HashMap<>();
        HashMap<INode, ArrayList<Outcome>> outcomeMap = networkHistory.getStateAtTimestep(timestep);
        for (Entry<INode, ArrayList<Outcome>> entry : outcomeMap.entrySet()) {
            ITrainable node = (ITrainable) entry.getKey();
            for(Outcome outcome : entry.getValue()){
                gradientMap.put(outcome, computeGradientOfOutcome(node, outcome));
            }
        }
        return gradientMap;
    }

    protected Vec computeGradientOfOutcome(ITrainable node, Outcome outcome) {
        Vec gradient = new DenseVector(linearizer.totalNumOfVariables);
        outcome.trainingData = gradient;

        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof InputNode) {
            return (Vec) outcome.trainingData;
        }

        for (int i = 0; i < outcome.allRootOutcomes.length; i++) {
            // get the contributions for each outcome;
            Outcome rootOutcome = outcome.allRootOutcomes[i];

            // root derivative component
            Vec root_gradient = (Vec) rootOutcome.trainingData;
            root_gradient = root_gradient.multiply(1/rootOutcome.probability);

            // distribution derivative
            IFilter filter = node.getIncomingConnectionFrom(rootOutcome.node).get().filter;
            double[] filter_derivative;

            // if the filter is not a part of the inclusion set, invert the probability
            if(((outcome.binary_string >> i) & 0b1) == 0){
                filter_derivative = filter.getNegatedLogarithmicDerivative(rootOutcome.activatedValue); 
            }
            else{
                filter_derivative = filter.getLogarithmicDerivative(rootOutcome.activatedValue); 
            }
            
            // add the derivative to the gradient
            root_gradient = linearizer.addToVector(filter, filter_derivative, root_gradient);

            // scale the root gradient and add to the total
            gradient.mutableAdd(root_gradient.multiply(outcome.probability));
        }

        return gradient;

    }

    
}
