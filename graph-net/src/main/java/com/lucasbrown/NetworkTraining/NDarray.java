package com.lucasbrown.NetworkTraining;

/**
 * n-dimensional array
 */
public class NDarray {
    
    private final int[] strides;
    private final int[] dimensions;
    private final double[] array;

    public NDarray(int[] dimensions)
    {
        this.strides = new int[dimensions.length];
        this.dimensions = new int[dimensions.length];

        this.strides[0] = 1;
        this.dimensions[0] = dimensions[0];
        int size = dimensions[0];

        for(int i=1;i<dimensions.length;++i) {
            this.strides[i] = this.strides[i - 1] * dimensions[i - 1];
            this.dimensions[i] = dimensions[i];
            size *= dimensions[i];
        }

        array = new double[size];
    }

    public int indexOf(int... coords)
    {
        int idx = coords[0];
        for(int i=1;i<coords.length;++i)
        {
            idx += coords[i] * strides[i];
        }
        return idx;
    }

    public double get(int idx)
    {
        return array[idx];
    }

    public double get(int[] coords)
    {
        return array[indexOf(coords)];
    }

    public void set(int idx, double value)
    {
        array[idx] = value;
    }
    
    public void set(int[] coords, double value)
    {
        array[indexOf(coords)] = value;
    }

}
