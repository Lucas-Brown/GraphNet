package com.lucasbrown.HelperClasses.FunctionalInterfaces;

@FunctionalInterface
public interface TriFunction<T, U, V, R> {
    public abstract R apply(T t, U u, V v);
}
