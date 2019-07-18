package xyz.onerous.modularnetwork.layer;

public abstract class Layer {
	protected int neurons;
	protected double[] neuronValues;
	
	public Layer(int neurons) {
		this.neurons = neurons;
		
		neuronValues = new double[neurons];
	}
	
	public int neurons() { return neurons; }
	
	public void setNeuronValues(double[] neuronValues) {
		this.neuronValues = neuronValues;
	}
	
	public double[] neuronValues() {
		return neuronValues;
	}
}
