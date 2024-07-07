package com.lucasbrown.NetworkTraining.History;

import java.util.ArrayList;

public interface IStateGenerator<T> {
    
    public ArrayList<T> getStateRecords();
}
