package com.lucasbrown.GraphNetwork.Global.Trainers;

import jsat.linear.Vec;

public interface ISolver {

    public Vec solve(Vec gradient);

}
