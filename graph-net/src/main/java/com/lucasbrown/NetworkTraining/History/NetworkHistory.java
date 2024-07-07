package com.lucasbrown.NetworkTraining.History;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

/**
 * Simple shorthand for how this object is used in maintaining the history of the Graph network through Outcome objects
 */
public class NetworkHistory extends History<Outcome, INode>{

    public NetworkHistory(IStateGenerator<INode> stateGenerator) {
        super(stateGenerator);
    }
    
}
