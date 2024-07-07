package com.lucasbrown.HelperClasses;

public class PassFailCounter {
    
    private int passes;
    private int total;

    public PassFailCounter(){
        reset();
    }

    public void pass(){
        passes++;
        total++;
    }

    public void fail(){
        total++;
    }

    public double getPassRate(){
        return ((double)passes)/total;
    }

    public void reset(){
        passes = 0;
        total = 0;
    }
}
