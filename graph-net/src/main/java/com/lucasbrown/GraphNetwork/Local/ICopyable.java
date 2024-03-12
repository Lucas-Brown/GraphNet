package com.lucasbrown.GraphNetwork.Local;

public  interface ICopyable<T> {
    
    /**
     * Returns a copy of the given type
     * @return
     */
    public abstract T copy();
}
