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
}
