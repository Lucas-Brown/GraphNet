package com.lucasbrown.NetworkTraining.NetworkDerivatives;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.Trainers.FilterLinearizer;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

/**
 * A naming abomination
 */
public class ForwardNetworkFilterGradient implements INetworkGradient{

    protected FilterLinearizer linearizer;
    protected NetworkHistory networkHistory;

    private ArrayList<HashMap<Outcome, Vec>> gradientsThroughTime;

    public ForwardNetworkFilterGradient(FilterLinearizer linearizer) {
        this.linearizer = linearizer;
    }

    @Override
    public ArrayList<HashMap<Outcome, Vec>> getGradient(NetworkHistory networkHistory) {
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
            INode node = entry.getKey();
            for(Outcome outcome : entry.getValue()){
                gradientMap.put(outcome, computeGradientOfOutcome(node, outcome));
            }
        }
        return gradientMap;
    }

    protected Vec computeGradientOfOutcome(INode node, Outcome outcome) {
        Vec gradient = new DenseVector(1);
        outcome.trainingData = gradient;

        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof IInputNode) {
            return gradient;
        }

        
        // if the probability is zero, then this contributes nothing to the final outcome
        if(outcome.probability == 0){
            return gradient;
        }

        int root_count = 0;
        int key = outcome.root_bin_str;
        IFilter[] filters = node.getProbabilityCombinator().getFilters(key);

        for (int i = 0; root_count < outcome.allRootOutcomes.length; i++) {
            if(((key >> i) & 0b1) == 0){
                continue;
            }

            // get the contributions for each outcome;
            Outcome rootOutcome = outcome.allRootOutcomes[root_count];

            // root derivative component
            Vec root_gradient = (Vec) rootOutcome.trainingData;
            root_gradient = root_gradient.divide(rootOutcome.probability);

            // distribution derivative
            IFilter filter = filters[root_count];
            double filter_derivative;

            // if the filter is not a part of the inclusion set, invert the probability
            if(((outcome.binary_string >> i) & 0b1) == 0){
                filter_derivative = filter.getNegatedLogarithmicDerivative(rootOutcome.activatedValue); 
            }
            else{
                filter_derivative = filter.getLogarithmicDerivative(rootOutcome.activatedValue); 
            }
            
            // add the derivative to the gradient
            root_gradient.mutableAdd(filter_derivative);

            
            for (double d : root_gradient.arrayCopy()) {
                assert Double.isFinite(d);
            }

            // scale the root gradient and add to the total
            gradient.mutableAdd(root_gradient);
            root_count++;
        }

        for (double d : gradient.arrayCopy()) {
            assert Double.isFinite(d);
        }
        gradient.mutableMultiply(outcome.probability);
        return gradient;

    }
    
}
