package com.lucasbrown.GraphNetwork.Global.Network;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

public class NodeBuilder {

    private final GraphNetwork network;

    private NodeConstructor nodeConstructor;
    private ActivationFunction activationFunction;
    private boolean is_input;
    private boolean is_output;

    private Supplier<ITrainableDistribution> outputDistributionSupplier;
    private Supplier<ITrainableDistribution> probabilityDistributionSupplier;

    private Function<ITrainableDistribution, IExpectationAdjuster> outputDistributionAdjusterSupplier;
    private Function<ITrainableDistribution, IExpectationAdjuster> probabilityDistributionAdjusterSupplier;

    private Exception buildFailureException;

    public NodeBuilder(final GraphNetwork network) {
        this.network = network;
    }

    public void setNodeConstructor(NodeConstructor nodeConstructor) {
        this.nodeConstructor = nodeConstructor;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public void setAsHiddenNode() {
        is_input = false;
        is_output = false;
    }

    public void setAsInputNode() {
        is_input = true;
        is_output = false;
    }

    public void setAsOutputNode() {
        is_output = true;
        is_input = false;
    }

    public void setOutputDistSupplier(Supplier<ITrainableDistribution> outputDistributionSupplier) {
        this.outputDistributionSupplier = outputDistributionSupplier;
    }

    public void setProbabilityDistSupplier(Supplier<ITrainableDistribution> probabilityDistributionSupplier) {
        this.probabilityDistributionSupplier = probabilityDistributionSupplier;
    }

    public void setOutputDistAdjusterSupplier(
            Function<ITrainableDistribution, IExpectationAdjuster> outputDistributionAdjusterSupplier) {
        this.outputDistributionAdjusterSupplier = outputDistributionAdjusterSupplier;
    }

    public void setProbabilityDistAdjusterSupplier(
            Function<ITrainableDistribution, IExpectationAdjuster> probabilityDistributionAdjusterSupplier) {
        this.probabilityDistributionAdjusterSupplier = probabilityDistributionAdjusterSupplier;
    }

    public boolean isReadyToBuild() {
        return nodeConstructor != null & activationFunction != null & outputDistributionSupplier != null
                & probabilityDistributionSupplier != null;
    }

    public Exception getBuildFailureException() {
        return buildFailureException;
    }

    /**
     * Attempts to build the node. Requires the node class, activation function,
     * output distribution, and probability distribution to all be set.
     * If one or more of these parameters has not been set, null will be returned
     * and an {@link IncompleteNodeException} will be stored as the failure
     * exception.
     * Similarly, if the node cannot be instantiated from the provided class object,
     * null will be returned and the corresponding error will stored as the failure
     * exception.
     * 
     * @return A new node with the provided parameters, or null if an exception was
     *         raised.
     * @see NodeBuilder#isReadyToBuild()
     * @see NodeBuilder#getBuildFailureException()
     */
    public INode build() {
        buildFailureException = null;

        if (!isReadyToBuild()) {
            buildFailureException = new IncompleteNodeException();
            return null;
        }

        ITrainableDistribution outputDistribution = outputDistributionSupplier.get();
        if (outputDistributionAdjusterSupplier == null) {
            outputDistributionAdjusterSupplier = outputDistribution.getDefaulAdjuster();
        }

        ITrainableDistribution probabilityDistribution = probabilityDistributionSupplier.get();
        if (probabilityDistributionAdjusterSupplier == null) {
            probabilityDistributionAdjusterSupplier = probabilityDistribution.getDefaulAdjuster();
        }

        INode node = nodeConstructor.apply(network, activationFunction, outputDistribution,
        outputDistributionAdjusterSupplier.apply(outputDistribution), probabilityDistribution,
        probabilityDistributionAdjusterSupplier.apply(probabilityDistribution));

        if (is_input) {
            node = new InputNode(node);
        } else if (is_output) {
            node = new OutputNode(node);
        }

        network.addNodeToNetwork(node);
        return node;

    }

    /**
     * Build N copies of the same node.
     * 
     * @param copies The number of copies to create
     * @return N copies
     * @see NodeBuilder#build()
     */
    public INode[] build(int copies) {
        return Stream.generate(this::build).limit(copies).toArray(INode[]::new);
    }


    
    // Shorter name for the quad-function in this context
    @FunctionalInterface
    public static interface NodeConstructor {

        public abstract INode apply(GraphNetwork network, final ActivationFunction activationFunction,
        ITrainableDistribution outputDistribution, IExpectationAdjuster outputAdjuster,
            ITrainableDistribution signalChanceDistribution, IExpectationAdjuster chanceAdjuster);
    }
}
