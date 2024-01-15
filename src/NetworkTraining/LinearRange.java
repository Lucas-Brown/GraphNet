package src.NetworkTraining;

public class LinearRange extends Range{
    
    private final double lowerBound;
    private final double upperBound;

    public LinearRange(double lowerBound, double upperBound, int n_divisions, boolean isLowerInclusive, boolean isUpperInclusive) 
    {
        super(n_divisions);
        int start = 0;
        int spacing = n_divisions - 1;

        if(!isLowerInclusive)
        {
            start += 1;
            spacing += 1;
        }
        

        if(!isUpperInclusive)
        {
            spacing += 1;
        }

        double delta = (upperBound - lowerBound)/spacing;
        for(int i = 0; i < n_divisions; i++)
        {
            values[i] = (i + start) * delta + lowerBound;
        }
        
        this.lowerBound = values[0];
        this.upperBound = values[n_divisions - 1];
    }

    private double getFloatingIndex(double x)
    {
        return (n_divisions - 1) * (x - lowerBound)/(upperBound - lowerBound);
    }

    @Override
    public int getNearestIndex(double x) 
    {
        double floatingIndex = getFloatingIndex(x);
        if(floatingIndex < 0 || floatingIndex >= n_divisions)
        {
            return -1;
        }
        else
        {
            return (int) floatingIndex; 
        }
    }

    @Override
    public double getIndexResidualWeight(double x) 
    {
        double floatingIndex = getFloatingIndex(x);
        return floatingIndex % 1.0;
    }
    
}
