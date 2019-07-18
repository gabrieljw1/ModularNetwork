package xyz.onerous.modularnetwork.layer;

import xyz.onerous.modularnetwork.ModularNetwork;

public abstract class Layer implements Cloneable {
	protected int neurons;
	protected double[] neuronValues;
	
	public Layer(int neurons) {
		this.neurons = neurons;
		
		neuronValues = new double[neurons];
	}
	
	public Layer(int neurons, double[] neuronValues) {
		this.neurons = neurons;
		this.neuronValues = neuronValues;
	}
	
	public int neurons() { return neurons; }
	
	public void setNeuronValues(double[] neuronValues) {
		this.neuronValues = neuronValues;
	}
	
	public double[] neuronValues() {
		return neuronValues;
	}
	
	@Override public Layer clone() {
		try {
			return (Layer)super.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			return null;
		}
	}
}
