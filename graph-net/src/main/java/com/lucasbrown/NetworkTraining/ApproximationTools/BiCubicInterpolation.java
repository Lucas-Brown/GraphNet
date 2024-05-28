package com.lucasbrown.NetworkTraining.ApproximationTools;

import java.util.function.ToDoubleBiFunction;

public class BiCubicInterpolation {

    private final double[] x_range;
    private final double[] y_range;
    private final ToDoubleBiFunction<Double, Double> generatingFunction;
    private final double[][] samplePoints;
    private final double[][][][] weights;

    public BiCubicInterpolation(double[] x_range, double[] y_range,
            final ToDoubleBiFunction<Double, Double> generatingFunction) {
        this.x_range = x_range;
        this.y_range = y_range;
        this.generatingFunction = generatingFunction;
        samplePoints = generateSamples();
        weights = computeWeights();
    }

    private double[][] generateSamples() {
        double[][] samples = new double[x_range.length][y_range.length];

        for (int i = 0; i < x_range.length; i++) {
            for (int j = 0; j < y_range.length; j++) {
                samples[i][j] = generatingFunction.applyAsDouble(x_range[i], y_range[j]);
            }
        }

        return samples;
    }
    private double[][][][] computeWeights() {
        double[][][][] w = new double[x_range.length - 3][y_range.length - 3][4][4];
    
        for (int i = 1; i < x_range.length - 2; i++) {
            for (int j = 1; j < y_range.length - 2; j++) {
    
                double[][] matrix = getSampleMatrix(i, j);

                for (int k = 0; k < 4; k++) {
                    for (int l = 0; l < 4; l++) {
                        double coeff = 0;
                        for (int m = 0; m < 4; m++) {
                            for (int n = 0; n < 4; n++) {
                                coeff += getMatrixCoef(k, m) * getMatrixCoef(l, n) * matrix[m][n]; 
                            }
                        }
                        w[i-1][j-1][k][l] = coeff;
                    }
                }
                
            }
        }
    
        return w;
    }
    
    
    // Helper function to get the value from the fixed matrix
    private double getMatrixCoef(int row, int col) {
        int[][] matrix = {
            { 1, 0, 0, 0 },
            { 0, 0, 1, 0 },
            { -3, 3, -2, -1 },
            { 2, -2, 1, 1 }
        };
        return matrix[row][col];
    }
    
    
    // Helper function to get the value from the sample matrix
    private double[][] getSampleMatrix(int i, int j) {
    
        // Compute finite differences for derivatives
        double fx_i_j = (samplePoints[i+1][j] - samplePoints[i-1][j]);
        double fx_i_j1 = (samplePoints[i+1][j] - samplePoints[i-1][j]);
        double fx_i1_j = (samplePoints[i+2][j] - samplePoints[i][j]);
        double fx_i1_j1 = (samplePoints[i+2][j+1] - samplePoints[i][j+1]);
        double fy_i_j = (samplePoints[i][j+1] - samplePoints[i][j-1]);
        double fy_i1_j = (samplePoints[i+1][j+1] - samplePoints[i+1][j-1]);
        double fy_i_j1 = (samplePoints[i][j+2] - samplePoints[i][j]);
        double fy_i1_j1 = (samplePoints[i+1][j+2] - samplePoints[i+1][j]);
        double fxy_i_j = fx_i_j + fy_i_j;
        double fxy_i1_j = fx_i1_j + fy_i1_j;
        double fxy_i_j1 = fx_i_j1 + fy_i_j1;
        double fxy_i1_j1 = fx_i1_j1 + fy_i1_j1;

        return new double[][]{
            {samplePoints[i][j], samplePoints[i][j+1], fy_i_j, fy_i_j1},
            {samplePoints[i+1][j], samplePoints[i+1][j+1], fy_i1_j, fy_i1_j1},
            {fx_i_j, fx_i_j1, fxy_i_j, fxy_i_j1},
            {fx_i1_j, fx_i1_j1, fxy_i1_j, fxy_i1_j1}
        };
    }

    /**
     * Interpolate the data at the point (x, y).
     * Extrapolation of points will use the generating function and will not expand
     * the evaluation range
     * 
     * @param x the x-component to interpolate
     * @param y the y-component to interpolate
     * @return The interpolated value
     */
    public double interpolate(double x, double y) {

        // Check if coordinates are within the sample range
        if (x < x_range[0] || x > x_range[x_range.length - 1] || y < y_range[0] || y > y_range[y_range.length - 1]) {
            // Extrapolation: use the generating function
            return generatingFunction.applyAsDouble(x, y); 
        }
    
        // Find the cell containing the point (x, y)
        int i = findIndex(x, x_range);
        int j = findIndex(y, y_range);
    
        // Ensure indices are within bounds for weights array
        i = Math.max(1, Math.min(i, weights.length - 1));
        j = Math.max(1, Math.min(j, weights[0].length - 1));
    
        // Calculate local coordinates within the cell
        double tx = (x - x_range[i]) / (x_range[i + 1] - x_range[i]);
        double ty = (y - y_range[j]) / (y_range[j + 1] - y_range[j]);
    
        // Evaluate polynomial using pre-computed weights
        double interpolatedValue = 0.0;
        for (int k = 0; k < 4; k++) {
            for (int l = 0; l < 4; l++) {
                interpolatedValue += weights[i - 1][j - 1][k][l] 
                                    * Math.pow(tx, k) 
                                    * Math.pow(ty, l);
            }
        }
        return interpolatedValue;
    }
    
    // Helper method to find the index of the cell
    private int findIndex(double value, double[] arr) {
        int idx = 0;
        while (idx < arr.length - 1 && value > arr[idx]) {
            idx++;
        }
        return idx;
    }

}
