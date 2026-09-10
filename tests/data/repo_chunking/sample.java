public class Calculator {
    private double result;
    
    public Calculator() {
        this.result = 0.0;
    }
    
    public double add(double a, double b) {
        this.result = a + b;
        return this.result;
    }
    
    public double subtract(double a, double b) {
        this.result = a - b;
        return this.result;
    }
    
    public double getResult() {
        return this.result;
    }
    
    public void reset() {
        this.result = 0.0;
    }
}
