package com.lucasbrown.HelperClasses;

public class MathHelpers {
    
    public static double clamp(double x, double min, double max){
        if(x < min){
            return min;
        }
        else if(x > max){
            return max;
        }
        else{
            return x;
        }
    }

    public static double sigmoid(double x){
        return 1/(1+Math.exp(-x));
    }

    public static double sigmoid_derivative(double x){
        return sigmoid(x)*sigmoid(-x);
    }
}
