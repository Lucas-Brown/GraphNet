package com.lucasbrown.GraphNetwork.Global;

import java.util.function.Supplier;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.Node;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.IProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;

public class NodeBuilder {

    private final GraphNetwork network;

    private ActivationFunction activationFunction;
    private boolean is_input;
    private boolean is_output;

    private Supplier<IValueCombinator> valueCombinator;
    private Supplier<IProbabilityCombinator> probabilityCombinator;

    public NodeBuilder(final GraphNetwork network) {
        this.network = network;
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

    public void setValueCombinator(Supplier<IValueCombinator> valueCombinator) {
        this.valueCombinator = valueCombinator;
    }

    public void setProbabilityCombinator(Supplier<IProbabilityCombinator> probabilityCombinator) {
        this.probabilityCombinator = probabilityCombinator;
    }

    public boolean isReadyToBuild() {
        return activationFunction != null && valueCombinator != null && probabilityCombinator != null;
    }

    public INode build() {
        if (!isReadyToBuild()) {
            throw new IncompleteNodeException();
        }

        INode node = new Node(network, activationFunction, valueCombinator.get(), probabilityCombinator.get());

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
}
