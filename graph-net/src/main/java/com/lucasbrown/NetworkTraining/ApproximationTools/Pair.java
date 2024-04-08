package com.lucasbrown.NetworkTraining.ApproximationTools;

public class Pair<U, V> {
    public U u;
    public V v;

    public U getU() {
        return u;
    }

    public void setU(U u) {
        this.u = u;
    }

    public V getV() {
        return v;
    }

    public void setV(V v) {
        this.v = v;
    }

    public Pair(U u, V v){
        this.u = u;
        this.v = v;
    }

}
