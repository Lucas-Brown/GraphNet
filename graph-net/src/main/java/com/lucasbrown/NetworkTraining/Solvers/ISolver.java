package com.lucasbrown.NetworkTraining.Solvers;

import jsat.linear.Vec;

public interface ISolver {

    public Vec solve(Vec gradient);

}
