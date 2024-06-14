package com.lucasbrown.GraphNetwork.Local.Nodes;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

public abstract class TrainableNodeBase extends NodeBase implements ITrainable{
 
    
    /**
     * The distribution of outputs produced by this node
     */
    protected ITrainableDistribution outputDistribution;

    /**
     * An objects which adjusts the parameters of outputDistribution given new data
     */
    protected IExpectationAdjuster outputAdjuster;

    /**
     * The probability distribution corresponding to signal passes
     */
    public ITrainableDistribution signalChanceDistribution;

    public IExpectationAdjuster chanceAdjuster;

    public TrainableNodeBase(GraphNetwork network, final ActivationFunction activationFunction,
            ITrainableDistribution outputDistribution,
            ITrainableDistribution signalChanceDistribution) {
        this(network, activationFunction, outputDistribution,
                outputDistribution.getDefaulAdjuster().apply(outputDistribution),
                signalChanceDistribution, signalChanceDistribution.getDefaulAdjuster().apply(signalChanceDistribution));
    }

    public TrainableNodeBase(GraphNetwork network, final ActivationFunction activationFunction,
            ITrainableDistribution outputDistribution, IExpectationAdjuster outputAdjuster,
            ITrainableDistribution signalChanceDistribution, IExpectationAdjuster chanceAdjuster){
            super(network, activationFunction);

            this.outputDistribution = outputDistribution;
            this.outputAdjuster = outputAdjuster;
            this.signalChanceDistribution = signalChanceDistribution;
            this.chanceAdjuster = chanceAdjuster;   
    }

    
    @Override
    public ITrainableDistribution getOutputDistribution() {
        return outputDistribution;
    }

    @Override
    public ITrainableDistribution getSignalChanceDistribution() {
        return signalChanceDistribution;
    }

    
    @Override
    public IExpectationAdjuster getOutputDistributionAdjuster() {
        return outputAdjuster;
    }

    @Override
    public IExpectationAdjuster getSignalChanceDistributionAdjuster() {
        return chanceAdjuster;
    }
    
    @Override
    public void applyDistributionUpdate() {
        outputAdjuster.applyAdjustments();
        chanceAdjuster.applyAdjustments();
    }

    @Override
    public void applyFilterUpdate() {
        for (Arc connection : outgoing) {
            if (connection.filterAdjuster != null) {
                connection.filterAdjuster.applyAdjustments();
            }
        }
    }
}
