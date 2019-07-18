package xyz.onerous.modularnetwork.datapackage;

/**
 * A package of many useful variables when learning, modifying the network, or just passing information
 * around. This is convenient due to the sheer number of different variables needed to be moved around when
 * performing different functions (when backpropogating, for example).
 * 
 * The network data package contains the neuron activations (a), neuron 'z's (z), neuron biases (b), 
 * connection weights (w), and the value of the connection (c).
 * 
 * @author Gabriel Wong
 */
public class NetworkDataPackage {
	private int networkResult;
	
	private double[][] a;
	private double[][] z;
	private double[][] b;
	
	private double[][][] w;
	
	//Getters for all variables.
	public int getNetworkResult() {
		return networkResult;
	}
	public double[][] getNeuronActivations() {
		return a;
	}
	public double[][] getNeuronZs() {
		return z;
	}
	public double[][] getNeuronBiases() {
		return b;
	}
	public double[][][] getConnectionWeights() {
		return w;
	}
	
	/**
	 * Create a new network data package with all of the following information.
	 * 
	 * @param networkResult
	 * @param neuronActivations
	 * @param neuronZs
	 * @param neuronBiases
	 * @param connectionWeights
	 * @param connectionValues
	 */
	public NetworkDataPackage(int networkResult, double[][] neuronActivations, double[][] neuronZs, double[][] neuronBiases, double[][][] connectionWeights) {
		this.networkResult = networkResult;
		this.a = neuronActivations;
		this.z = neuronZs;
		this.b = neuronBiases;
		this.w = connectionWeights;
	}
	
	public double[][][] getConnectionValues() {
		double[][][] connectionValues = w.clone();
		
		for (int l = 1; l < connectionValues.length; l++) {
			for (int n = 0; n < connectionValues[l].length; n++) {
				for (int c = 0; c < connectionValues[l][n].length; c++) {
					connectionValues[l][n][c] = a[l-1][c] * w[l][n][c];
				}
			}
		}
		
		return connectionValues;
	}
}
