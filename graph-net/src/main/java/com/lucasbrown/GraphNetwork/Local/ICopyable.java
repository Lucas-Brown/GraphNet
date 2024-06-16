package com.lucasbrown.GraphNetwork.Local;

import java.util.Collection;

public interface ICopyable<T> {

    /**
     * Returns a copy of the given type
     * 
     * @return
     */
    public abstract T copy();

    public static <E extends ICopyable<E>> void collectionCopy(Collection<E> src, Collection<E> dst) {
        for (E e : src) {
            dst.add(e.copy());
        }
    }

    @SuppressWarnings("unchecked")
    public static <E, F extends E> void collectionCopyUnsafe(Collection<E> src, Collection<E> dst) {
        for (E e : src) {
            dst.add((E) ((ICopyable<F>) e).copy());
        }
    }
}
