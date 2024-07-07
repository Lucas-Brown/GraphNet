package com.lucasbrown.NetworkTraining.NetworkDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;

import jsat.linear.Vec;

public interface INetworkGradient {
    
    public ArrayList<HashMap<Outcome, Vec>> getGradient(NetworkHistory networkHistory);

}
