package xyz.onerous.modularnetwork.layer.activationlayer;

import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.layer.Layer;
import xyz.onerous.modularnetwork.type.ActivationType;

public abstract class ActivationLayer extends Layer {
	//Activation type information
	protected ActivationType activationType;
	
	//Dimensions
	protected int pNeurons;
	protected double[] bias;
	protected double[][] weight;
	protected double[] weightedInput;
	
	//Working variables
	protected boolean hasBeenInitialized = false;
	
	public ActivationType activationType() {
		return activationType;
	}
	
	public double[] weightedInput() {
		return weightedInput;
	}
	
	public double[] bias() {
		return bias;
	}
	
	public void setBias(double[] bias) {
		this.bias = bias;
	}
	
	public double[][] weight() {
		return weight;
	}
	
	public void setWeight(double[][] weight) {
		this.weight = weight;
	}
	
	public ActivationLayer(ActivationType activationType, int neurons, int pNeurons) {
		super(neurons);
		
		this.activationType = activationType;
		this.pNeurons = pNeurons;
		
		initialize();
	}
	
	public ActivationLayer(ActivationType activationType, int neurons, int pNeurons, double[] bias, double[][] weight) throws InvalidInputLengthException {
		super(neurons);
		
		this.activationType = activationType;
		this.pNeurons = pNeurons;
		
		initialize(bias, weight);
	}
	
	/**
	 * Called to fill out the bias and weight arrays with the proper spaces
	 */
	private void initialize() {
		if (hasBeenInitialized) { return; }
		
		bias = new double[neurons];
		weight = new double[neurons][];
		
		for (int i = 0; i < neurons; i++) {
			weight[i] = new double[pNeurons];
		}
		
		weightedInput = bias.clone();
		
		hasBeenInitialized = true;
	}
	
	private void initialize(double[] bias, double[][] weight) throws InvalidInputLengthException {
		if (hasBeenInitialized) { return; }
		
		this.bias = bias;
		this.weight = weight;
		
		if (bias.length != neurons) {
			throw new InvalidInputLengthException("bias fault");
		}
		
		for (int i = 0; i < neurons; i++) {
			if (weight[i].length != pNeurons) {
				throw new InvalidInputLengthException("weight fault");
			}
		}
		
		weightedInput = bias.clone();
		
		hasBeenInitialized = true;
	}
	
	public double[] calculateWeightedInput(double[] input) {
		double[] weightedInput = new double[neurons];
		
		for (int i = 0; i < neurons; i++) {
			weightedInput[i] = 0;
			
			for (int j = 0; j < pNeurons; j++) {
				weightedInput[i] += input[j]*weight[i][j];
			}
		}
		
		return weightedInput;
	}
	
	public void setWeight(int n, int pN, double weightValue) {
		weight[n][pN] = weightValue;
	}
	
	public void setBias(int n, double biasValue) {
		bias[n] = biasValue;
	}
	
	public static Layer generateActivationLayer(ActivationType activationType, int neurons, int pNeurons) {
		switch (activationType) {
		case Linear:
			return new LinearLayer(neurons, pNeurons);
		case ReLU:
			return new ReLULayer(neurons, pNeurons);
		case Sigmoid:
			return new SigmoidLayer(neurons, pNeurons);
		case TanH:
			return new TanHLayer(neurons, pNeurons);
		case LeakyReLU:
			break;
		default:
			break;
		}
		
		return null;
	}
	
	public double[] activate(double[] input) throws InvalidInputLengthException {
		if (input.length != pNeurons) { throw new InvalidInputLengthException(); }
		
		weightedInput = calculateWeightedInput(input);
		double[] activation = new double[neurons];
		
		for (int i = 0; i < neurons; i++) {			
			activation[i] = activationFunction(weightedInput[i]);
		}
		
		setNeuronValues(activation);
		
		return activation;
	}
	
	public double[] activatePrime(double[] input) throws InvalidInputLengthException {
		if (input.length != pNeurons) { throw new InvalidInputLengthException(); }
		
		weightedInput = calculateWeightedInput(input);
		double[] activationPrime = new double[neurons];
		
		for (int i = 0; i < neurons; i++) {			
			activationPrime[i] = activationPrimeFunction(weightedInput[i]);
		}
		
		return activationPrime;
	}
	
	public double[] softmaxLayer() {
		double layerSum = 0.0;
		double[] softmaxValues = new double[neurons];
		
		for (double neuronZ : weightedInput) {
			layerSum += Math.exp(neuronZ);
		}
		
		for (int i = 0; i < neurons; i++) {
			softmaxValues[i] = Math.exp(weightedInput[i]) / layerSum;
		}

		return softmaxValues;
	}
	
	public double[] softmaxPrime() {
		double[] softmaxPrimeValues = new double[neurons];
		double[] softmaxValues = softmaxLayer();
		
		for (int i = 0; i < neurons; i++) {
			softmaxPrimeValues[i] = softmaxValues[i] * (1 - softmaxValues[i]);
		}
		
		return softmaxPrimeValues;
	}
	
	protected abstract double activationFunction(double weightedInput);
	protected abstract double activationPrimeFunction(double weightedInput);
}
