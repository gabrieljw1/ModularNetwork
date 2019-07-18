package xyz.onerous.modularnetwork.learner.multithread;

import xyz.onerous.modularnetwork.ModularNetwork;
import xyz.onerous.modularnetwork.datapackage.WeightBiasDeltaPackage;
import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.layer.activationlayer.ActivationLayer;
import xyz.onerous.modularnetwork.type.ActivationType;
import xyz.onerous.modularnetwork.type.LossType;
import xyz.onerous.modularnetwork.util.MatrixUtil;

public class MTTrainThread extends Thread {
	private Thread t;
	private String threadName;
	
	protected ModularNetwork modularNetwork;
	
	private double[] trainData;
	private int expectedOutput;
	
	private LossType lossType;
	private boolean usingSoftmax;
	private double learningRate;
	
	protected WeightBiasDeltaPackage deltaPackage;
	
	MTTrainThread(ModularNetwork modularNetwork, double[] trainData, int expectedOutput, LossType lossType, boolean usingSoftmax, double learningRate, String threadName) {
		this.threadName = threadName;
		
		this.modularNetwork = modularNetwork.clone();
		
		this.trainData = trainData;
		this.expectedOutput = expectedOutput;
		
		this.lossType = lossType;
		this.usingSoftmax = usingSoftmax;
		this.learningRate = learningRate;
	}
	
	@Override
	public void run() {
		try {
			deltaPackage = performTrainAndGetDelta();
		} catch (InvalidInputLengthException e) {
			e.printStackTrace();
		}
	}
	
	@Override
	public void start() {
		if (t == null) {
			t = new Thread(this, threadName);
			t.start();
		}
	}
	
	public WeightBiasDeltaPackage gradientDescent(int expectedIndex, LossType lossType, boolean usingSoftmax, double learningRate) { 
		//Batch size needed to limit weight/bias changing over an entire batch
		double[][] δ = new double[modularNetwork.layers.size()][];

		for (int l = 0; l < modularNetwork.layers.size(); l++) {
			δ[l] = new double[ modularNetwork.layers.get(l).neurons() ];
		}
	
		
		//CALCULATE OUTPUT LAYER FIRST
		double[] expectedOutput = new double[modularNetwork.layers.get(modularNetwork.layers.size()-1).neurons()];
		
		expectedOutput[expectedIndex] = 1.0;
		
		if (((ActivationLayer)modularNetwork.layers.get(modularNetwork.layers.size()-1)).activationType() == ActivationType.TanH) {
			for (int i = 0; i < expectedOutput.length; i++) {
				if (i != expectedIndex) {
					expectedOutput[i] = -1.0;
				}
			}
		}
		
		switch (lossType) {
		case MeanSquaredError: //  (actual - predicted)^2 / n    so the deriv is    (2/n)(actual-predicted)
			for (int i = 0; i < modularNetwork.layers.get(modularNetwork.layers.size()-1).neurons(); i++) {
				δ[modularNetwork.layers.size()-1][i] = 0.5 * (modularNetwork.layers.get(modularNetwork.layers.size()-1).neuronValues()[i] - expectedOutput[i]);
			}
			break;
		case MeanAbsoluteError:
			for (int i = 0; i < modularNetwork.layers.get(modularNetwork.layers.size()-1).neurons(); i++) {
				//if (modularNetwork.layers.get(modularNetwork.layers.size()-1).neuronValues()[i] > expectedOutput[i]) {
				if (modularNetwork.layers.get(modularNetwork.layers.size()-1).neuronValues()[i] > expectedOutput[i]) {
					δ[modularNetwork.layers.size()-1][i] = +1.0;
				} else if (modularNetwork.layers.get(modularNetwork.layers.size()-1).neuronValues()[i] < expectedOutput[i]) {
					δ[modularNetwork.layers.size()-1][i] = -1.0;
				} else {
					δ[modularNetwork.layers.size()-1][i] = +0.0;
				}
			}
			break;
		case CrossEntropy:
			for (int i = 0; i < modularNetwork.layers.get(modularNetwork.layers.size()-1).neurons(); i++) {
				δ[modularNetwork.layers.size()-1][i] = (-expectedOutput[i]/modularNetwork.layers.get(modularNetwork.layers.size()-1).neuronValues()[i]) + (1.0 - expectedOutput[i])/(1.0 - modularNetwork.layers.get(modularNetwork.layers.size()-1).neuronValues()[i]);
			}
			break;
		case BinaryCrossEntropy:
			//TODO: Implement
			break;
		default: 
			System.out.println("Default switch thrown at δ calculation"); 
			break;
		}
		
		
		//Up until now, only part of δ has been stored inside.
		//We still have to hadamard the delCdelA with the derivative of the activation function for z.
		if (usingSoftmax) {
			δ[modularNetwork.layers.size()-1] = MatrixUtil.hadamard(δ[modularNetwork.layers.size()-1], (((ActivationLayer)modularNetwork.layers.get(modularNetwork.layers.size()-1)).softmaxPrime()));
		} else {
			δ[modularNetwork.layers.size()-1] = MatrixUtil.hadamard(δ[modularNetwork.layers.size()-1], ((ActivationLayer)modularNetwork.layers.get(modularNetwork.layers.size()-1)).neuronValues());
		}
		
		
		//
		//Calculate rest of network
		//
		for (int l = modularNetwork.layers.size() - 2; l > 0; l--) { //activate layer prime
			try {
				δ[l] = MatrixUtil.hadamard(
						MatrixUtil.multiplyWithFirstTranspose(
								((ActivationLayer)modularNetwork.layers.get(l+1)).weight(), 
								δ[l+1]), 
							((ActivationLayer)modularNetwork.layers.get(l)).activatePrime((modularNetwork.layers.get(l-1).neuronValues())
						)
					);
			} catch (InvalidInputLengthException e) {
				e.printStackTrace();
			}
		}
		
		
		//Gradient Descent
		double[][][] deltaW = new double[modularNetwork.layers.size()][][];
		deltaW[0] = new double[0][];
		for (int l = 1; l < modularNetwork.layers.size(); l++) {
			deltaW[l] = new double [modularNetwork.layers.get(l).neurons()][];
			for (int n = 0; n < modularNetwork.layers.get(l).neurons(); n++) {
				deltaW[l][n] = new double[modularNetwork.layers.get(l-1).neurons()];
			}
		}
				
		double[][] deltaB = δ.clone();
		
		for (int l = modularNetwork.layers.size() - 1; l > 0; l--) {
			deltaW[l] = MatrixUtil.scalarMultiply(MatrixUtil.multiplyWithSecondTranspose(δ[l], modularNetwork.layers.get(l-1).neuronValues()), -learningRate);
			deltaB[l] = MatrixUtil.scalarMultiply(δ[l], -learningRate);
		}
		
		return new WeightBiasDeltaPackage(deltaW, deltaB);
	}
	
	public WeightBiasDeltaPackage performTrainAndGetDelta() throws InvalidInputLengthException {
		modularNetwork.activateNetwork(trainData);

		return gradientDescent(expectedOutput, lossType, usingSoftmax, learningRate);
	}
	
}
