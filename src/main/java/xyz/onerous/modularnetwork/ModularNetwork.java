package xyz.onerous.modularnetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import xyz.onerous.modularnetwork.util.ArrayUtil;
import xyz.onerous.modularnetwork.util.MatrixUtil;
import xyz.onerous.modularnetwork.datapackage.NetworkDataPackage;
import xyz.onerous.modularnetwork.datapackage.TestResultPackage;
import xyz.onerous.modularnetwork.datapackage.WeightBiasDeltaPackage;
import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.layer.Layer;
import xyz.onerous.modularnetwork.layer.activationlayer.ActivationLayer;
import xyz.onerous.modularnetwork.layer.inputlayer.InputLayer;
import xyz.onerous.modularnetwork.learner.BatchTrainer;
import xyz.onerous.modularnetwork.learner.multithread.MTBatchTrainer;
import xyz.onerous.modularnetwork.learner.singlethread.STBatchTrainer;
import xyz.onerous.modularnetwork.type.ActivationType;
import xyz.onerous.modularnetwork.type.LossType;
import xyz.onerous.modularnetwork.type.WeightInitType;

public class ModularNetwork implements Cloneable {
	public ArrayList<Layer> layers;
	
	double[] lastNetworkOutput;
	
	//Backpropagation variables
	boolean backPropagationVariablesInitialized = false;
	double[][] δ;
	
	//Modifiers
	public boolean multithreadingEnable = false;
	
	
	public ModularNetwork(int inputNeurons) {
		layers = new ArrayList<Layer>();
		
		layers.add(new InputLayer(inputNeurons));
	}
	
	public ModularNetwork(ArrayList<Layer> layers) {
		this.layers = layers;
	}
	
	public void addLayer(ActivationType activationType, int neurons) {
		int pNeurons = layers.get(layers.size() - 1).neurons();

		layers.add(ActivationLayer.generateActivationLayer(activationType, neurons, pNeurons));
	}
	
	public double[] activateNetwork(double[] input) throws InvalidInputLengthException {
		if (input.length != layers.get(0).neurons()) { throw new InvalidInputLengthException(); }
		
		layers.get(0).setNeuronValues(input);
		
		return activateLayer(layers.size()-1);
	}
	
	public int activateNetworkAndGetOutput(double[] input) throws InvalidInputLengthException {
		if (input.length != layers.get(0).neurons()) { throw new InvalidInputLengthException(); }
		
		layers.get(0).setNeuronValues(input);
		
		lastNetworkOutput = activateLayer(layers.size()-1);
		
		double maxValue = lastNetworkOutput[0];
		int maxValueAtIndex = 0;
		
		for (int n = 1; n < layers.get(layers.size()-1).neurons(); n++) {
			if (lastNetworkOutput[n] > maxValue) {
				maxValueAtIndex = n;
				maxValue = lastNetworkOutput[n];
			}
		}
		
		return maxValueAtIndex;
	}
	
	public int getOutput() {
		lastNetworkOutput = layers.get(layers.size()-1).neuronValues();
		
		double maxValue = lastNetworkOutput[0];
		int maxValueAtIndex = 0;
		
		for (int n = 1; n < layers.get(layers.size()-1).neurons(); n++) {
			if (lastNetworkOutput[n] > maxValue) {
				maxValueAtIndex = n;
				maxValue = lastNetworkOutput[n];
			}
		}
		
		return maxValueAtIndex;

	}
	
	public double[] activateLayer(int layerIndex) throws InvalidInputLengthException {
		if (layerIndex == 0) {
			return layers.get(0).neuronValues();
		} 
		
		double[] pActivations = activateLayer(layerIndex - 1);
		
		return ((ActivationLayer)layers.get(layerIndex)).activate(pActivations);
	}
	
	public void initializeWeightsOneType(WeightInitType weightInitType) {
		Random rand = new Random();
		
		switch (weightInitType) {
		case Kaiming:
			for (int l = 1; l < layers.size(); l++) {
				for (int n = 0; n < layers.get(l).neurons(); n++) {
					for (int pN = 0; pN < layers.get(l-1).neurons(); pN++) {
						double weightValue = rand.nextGaussian() * Math.sqrt(2.0 / layers.get(l-1).neurons());
						
						((ActivationLayer)layers.get(l)).setWeight(n, pN, weightValue);
					}
				}
			}
			break;
		case RandomNormal:
			for (int l = 1; l < layers.size(); l++) {
				for (int n = 0; n < layers.get(l).neurons(); n++) {
					for (int pN = 0; pN < layers.get(l-1).neurons(); pN++) {
						double weightValue = rand.nextGaussian();
						
						((ActivationLayer)layers.get(l)).setWeight(n, pN, weightValue);
					}
				}
			}
			break;
		case Xavier:
			for (int l = 1; l < layers.size(); l++) {
				for (int n = 0; n < layers.get(l).neurons(); n++) {
					for (int pN = 0; pN < layers.get(l-1).neurons(); pN++) {
						double weightValue = rand.nextGaussian() * Math.sqrt(1.0 / layers.get(l-1).neurons());
						
						((ActivationLayer)layers.get(l)).setWeight(n, pN, weightValue);
					}
				}
			}
			break;
		}
	}
	
	public void initializeWeightsDynamic() {
		for (int l = 1; l < layers.size(); l++) {
			switch (((ActivationLayer)layers.get(l)).activationType()) {
			case LeakyReLU:
				initializeLayerWeights(l, WeightInitType.Kaiming);
				break;
			case Linear:
				initializeLayerWeights(l, WeightInitType.Kaiming);
				break;
			case ReLU:
				initializeLayerWeights(l, WeightInitType.Kaiming);
				break;
			case Sigmoid:
				initializeLayerWeights(l, WeightInitType.Xavier);
				break;
			case TanH:
				initializeLayerWeights(l, WeightInitType.Xavier);
				break;
			}
		}
	}
	
	private void initializeLayerWeights(int layerIndex, WeightInitType weightInitType) {
		Random rand = new Random();
		
		switch (weightInitType) {
		case Kaiming:
				for (int n = 0; n < layers.get(layerIndex).neurons(); n++) {
					for (int pN = 0; pN < layers.get(layerIndex-1).neurons(); pN++) {
						double weightValue = rand.nextGaussian() * Math.sqrt(2.0 / layers.get(layerIndex-1).neurons());
						
						((ActivationLayer)layers.get(layerIndex)).setWeight(n, pN, weightValue);
					}
				}
			break;
		case RandomNormal:
				for (int n = 0; n < layers.get(layerIndex).neurons(); n++) {
					for (int pN = 0; pN < layers.get(layerIndex-1).neurons(); pN++) {
						double weightValue = rand.nextGaussian();
						
						((ActivationLayer)layers.get(layerIndex)).setWeight(n, pN, weightValue);
					}
				}
			break;
		case Xavier:
				for (int n = 0; n < layers.get(layerIndex).neurons(); n++) {
					for (int pN = 0; pN < layers.get(layerIndex-1).neurons(); pN++) {
						double weightValue = rand.nextGaussian() * Math.sqrt(1.0 / layers.get(layerIndex-1).neurons());
						
						((ActivationLayer)layers.get(layerIndex)).setWeight(n, pN, weightValue);
					}
				}
			break;
		}
	}
	
	public void applyConstantBias(double bias) {
		for (int l = 1; l < layers.size(); l++) {
			for (int n = 0; n < layers.get(l).neurons(); n++) {
				((ActivationLayer)layers.get(l)).setBias(n, bias);
			}
		}
	}
	
	private void initializeBackPropagationVariables() {
		δ = new double[layers.size()][];

		for (int l = 0; l < layers.size(); l++) {
			δ[l] = new double[ layers.get(l).neurons() ];
		}
		
		backPropagationVariablesInitialized = true;
	}
	
	private double getOutputError(int expectedIndex, LossType lossType) {
		double[] expectedOutput = new double[layers.get(layers.size()-1).neurons()];
		
		expectedOutput[expectedIndex] = 1.0;
		
		switch (lossType) {
		case MeanSquaredError: //  (actual - predicted)^2 / n
			double sumSquaredError = 0.0;
			for (int i = 0; i < layers.get(layers.size()-1).neurons(); i++) {
				sumSquaredError += Math.pow(layers.get(layers.size()-1).neuronValues()[i] - expectedOutput[i], 2.0);
			}
			return sumSquaredError / (double)(layers.get(layers.size()-1).neurons());
		case MeanAbsoluteError:
			double sumAbsoluteError = 0.0;
			for (int i = 0; i < layers.get(layers.size()-1).neurons(); i++) {
				sumAbsoluteError += Math.abs(layers.get(layers.size()-1).neuronValues()[i] - expectedOutput[i]);
			}
			return sumAbsoluteError / (double)(layers.get(layers.size()-1).neurons());
		case CrossEntropy:
			double totalCrossEntropyError = 0.0;
			for (int i = 0; i < layers.get(layers.size()-1).neurons(); i++) {
				totalCrossEntropyError += expectedOutput[i]*Math.log(layers.get(layers.size()-1).neuronValues()[i]);
			}
			return -totalCrossEntropyError;
		case BinaryCrossEntropy:
			return 0.0; //TODO: IMPLEMENT
		default: 
			System.out.println("Default switch thrown at network error calculation"); 
			return 0.0;
		}
	}

	private void backPropagate(int expectedIndex, LossType lossType, boolean usingSoftmax) {
		if (!this.backPropagationVariablesInitialized) { this.initializeBackPropagationVariables(); }
		
		//CALCULATE OUTPUT LAYER FIRST
		double[] expectedOutput = new double[layers.get(layers.size()-1).neurons()];
		
		expectedOutput[expectedIndex] = 1.0;
		
		if (((ActivationLayer)layers.get(layers.size()-1)).activationType() == ActivationType.TanH) {
			for (int i = 0; i < expectedOutput.length; i++) {
				if (i != expectedIndex) {
					expectedOutput[i] = -1.0;
				}
			}
		}
		
		switch (lossType) {
		case MeanSquaredError: //  (actual - predicted)^2 / n    so the deriv is    (2/n)(actual-predicted)
			for (int i = 0; i < layers.get(layers.size()-1).neurons(); i++) {
				δ[layers.size()-1][i] = 0.5 * (layers.get(layers.size()-1).neuronValues()[i] - expectedOutput[i]);
			}
			break;
		case MeanAbsoluteError:
			for (int i = 0; i < layers.get(layers.size()-1).neurons(); i++) {
				//if (layers.get(layers.size()-1).neuronValues()[i] > expectedOutput[i]) {
				if (layers.get(layers.size()-1).neuronValues()[i] > expectedOutput[i]) {
					δ[layers.size()-1][i] = +1.0;
				} else if (layers.get(layers.size()-1).neuronValues()[i] < expectedOutput[i]) {
					δ[layers.size()-1][i] = -1.0;
				} else {
					δ[layers.size()-1][i] = +0.0;
				}
			}
			break;
		case CrossEntropy:
			for (int i = 0; i < layers.get(layers.size()-1).neurons(); i++) {
				δ[layers.size()-1][i] = (-expectedOutput[i]/layers.get(layers.size()-1).neuronValues()[i]) + (1.0 - expectedOutput[i])/(1.0 - layers.get(layers.size()-1).neuronValues()[i]);
			}
			break;
		case BinaryCrossEntropy:
			
			break;
		default: 
			System.out.println("Default switch thrown at δ calculation"); 
			break;
		}
		
		
		//Up until now, only part of δ has been stored inside.
		//We still have to hadamard the delCdelA with the derivative of the activation function for z.
		if (usingSoftmax) {
			δ[layers.size()-1] = MatrixUtil.hadamard(δ[layers.size()-1], (((ActivationLayer)layers.get(layers.size()-1)).softmaxPrime()));
		} else {
			δ[layers.size()-1] = MatrixUtil.hadamard(δ[layers.size()-1], ((ActivationLayer)layers.get(layers.size()-1)).neuronValues());
		}
		
		
		//
		//Calculate rest of network
		//
		for (int l = layers.size() - 2; l > 0; l--) { //activate layer prime
			try {
				δ[l] = MatrixUtil.hadamard(
						MatrixUtil.multiplyWithFirstTranspose(
								((ActivationLayer)layers.get(l+1)).weight(), 
								δ[l+1]), 
							((ActivationLayer)layers.get(l)).activatePrime((layers.get(l-1).neuronValues())
						)
					);
			} catch (InvalidInputLengthException e) {
				e.printStackTrace();
			}
		}
	}
	
	public WeightBiasDeltaPackage gradientDescent(int expectedIndex, LossType lossType, boolean usingSoftmax, double learningRate) { //Batch size needed to limit weight/bias changing over an entire batch
		backPropagate(expectedIndex, lossType, usingSoftmax);
		
		double[][][] deltaW = new double[layers.size()][][];
		deltaW[0] = new double[0][];
		for (int l = 1; l < layers.size(); l++) {
			deltaW[l] = new double [layers.get(l).neurons()][];
			for (int n = 0; n < layers.get(l).neurons(); n++) {
				deltaW[l][n] = new double[layers.get(l-1).neurons()];
			}
		}
				
		double[][] deltaB = δ.clone();
		
		for (int l = layers.size() - 1; l > 0; l--) {
			deltaW[l] = MatrixUtil.scalarMultiply(MatrixUtil.multiplyWithSecondTranspose(δ[l], layers.get(l-1).neuronValues()), -learningRate);
			deltaB[l] = MatrixUtil.scalarMultiply(δ[l], -learningRate);
		}
		
		return new WeightBiasDeltaPackage(deltaW, deltaB);
	}
	
	public WeightBiasDeltaPackage performTrainAndGetDelta(double[] trainData, int expectedOutput, LossType lossType, boolean usingSoftmax, double learningRate) throws InvalidInputLengthException {
		activateNetwork(trainData);
		return gradientDescent(expectedOutput, lossType, usingSoftmax, learningRate);
	}
	
	/**
	 * Perform one training batch consisting of many single training iterations whose weight-bias deltas are
	 * summed and applied in one go. This is different than running many single training iterations by
	 * themselves because deltas are not applied after every iteration.
	 * 
	 * Batch training is sometimes more efficient than 'online' training (applying deltas after each
	 * iteration) operation-wise.
	 * 
	 * @param batchData Data to be inputted
	 * @param expectedOutputs The expected results of the network per the batch data
	 * @return The combined weight-bias deltas generated by the training iterations
	 * @throws InvalidInputLengthException 
	 */
	public WeightBiasDeltaPackage performBatchAndGetDelta(double[][] batchData, int[] expectedOutputs, LossType lossType, boolean usingSoftmax, double learningRate) throws InvalidInputLengthException {
		BatchTrainer batchTrainer;
		
		if (!multithreadingEnable) {
			batchTrainer = new STBatchTrainer();
		} else {
			batchTrainer = new MTBatchTrainer();
		}
		
		return batchTrainer.performBatch(this, batchData, expectedOutputs, lossType, usingSoftmax, learningRate);
	}
	
	/**
	 * Perform an entire epoch of training. An epoch is when the network is trained through the entire data
	 * set once. If batch training is not desired, a batchSize of one (1) can be specified. If batch training
	 * is desired, the epoch trainer will split the training data into batches. Should the training data not 
	 * divide evenly by the batch size, the last batch will have less training data than the previous ones.
	 * 
	 * @param trainingData Data to be inputted
	 * @param expectedOutputs The expected results of the network per the training data
	 * @param batchSize Size of the batches to be performed. 1 if batch training is not desired.
	 * @throws InvalidInputLengthException 
	 */
	public void performEpoch(double[][] trainingData, int[] expectedOutputs, int batchSize, LossType lossType, boolean usingSoftmax, double learningRate) throws InvalidInputLengthException {
		//Translate numbers into readable variables for code readability
		int numDataPoints = trainingData.length - 50000;
		int numBatches = (int)Math.ceil(numDataPoints / batchSize);

		
		//Find the individual batch sizes (accounting for if num batches does not divide evenly into training data)
		int[] batchSizes = new int[numBatches];
		
		if (numDataPoints % numBatches == 0) {
			for (int i = 0; i < numBatches; i++) {
				batchSizes[i] = batchSize;
			}
		} else {
			for (int i = 0; i < numBatches - 1; i++) {
				batchSizes[i] = batchSize;
			}
			
			batchSizes[numBatches - 1] = numDataPoints - ((numBatches - 1) * batchSize);
		}
		
		
		//Find what chunk of data each batch will have
		double[][][] batchDataSets = new double[numBatches][][];
		int[][] batchExpectedOutputs = new int[numBatches][];
		int nextBatchStartIndex = 0;
		
		for (int b = 0; b < numBatches; b++) {
			batchDataSets[b] = ArrayUtil.clipArray(trainingData, nextBatchStartIndex, nextBatchStartIndex+batchSizes[b]);
			batchExpectedOutputs[b] = ArrayUtil.clipArray(expectedOutputs, nextBatchStartIndex, nextBatchStartIndex+batchSizes[b]);
			nextBatchStartIndex += batchSizes[b];
		}
		
		
		//Run each batch and apply deltas
		for (int b = 0; b < numBatches; b++) {
			System.out.println("\nStarting batch " + b);
			applyDeltaPackage(performBatchAndGetDelta(batchDataSets[b], batchExpectedOutputs[b], lossType, usingSoftmax, learningRate));
		}
	}
	
	public void applyDeltaPackage(WeightBiasDeltaPackage deltaPackage) {
		for (int l = layers.size() - 1; l > 0; l--) {
			((ActivationLayer)layers.get(l)).setWeight(  MatrixUtil.add(((ActivationLayer)layers.get(l)).weight(), deltaPackage.deltaW[l])  );
			((ActivationLayer)layers.get(l)).setBias(  MatrixUtil.add(((ActivationLayer)layers.get(l)).bias(), deltaPackage.deltaB[l])  );
		}
	}
	
	public TestResultPackage performTest(double[][] testData, int[] expectedOutputs) throws InvalidInputLengthException {
		if (testData.length != expectedOutputs.length) { return (TestResultPackage) null; }
		
		int numTests = testData.length;
		int correctCount = 0;
		
		double[]  outputNeuronValues = new double[numTests];
		int[]     outputNeuronIndeces = new int[numTests];
		boolean[] ifCorrect = new boolean[numTests];
		
		for (int t = 0; t < numTests; t++) {
			int outputIndex = activateNetworkAndGetOutput(testData[t]);
			if (outputIndex == expectedOutputs[t]) { 
				correctCount++; 
				ifCorrect[t] = true;
			} else {
				ifCorrect[t] = false;
			}
			
			outputNeuronValues[t] = layers.get(layers.size()-1).neuronValues()[outputIndex]; 
			outputNeuronIndeces[t] = outputIndex;
		}
		
		double percentageCorrect = (double)correctCount / (double)numTests;
		
		return new TestResultPackage(numTests, percentageCorrect, outputNeuronValues, outputNeuronIndeces, ifCorrect);
	}
	
	public NetworkDataPackage generateNetworkDataPackage() {
		double[][] a = new double[layers.size()][];
		double[][] z = new double[layers.size()][];
		double[][] b = new double[layers.size()][];
		double[][][] w = new double[layers.size()][][];
		
		a[0] = layers.get(0).neuronValues();
		z[0] = a[0];
		b[0] = new double[0];
		w[0] = new double[0][];
		
		for (int l = 1; l < layers.size(); l++) {
			a[l] = ((ActivationLayer)layers.get(l)).neuronValues();
			z[l] = ((ActivationLayer)layers.get(l)).weightedInput();
			b[l] = ((ActivationLayer)layers.get(l)).bias();
			w[l] = ((ActivationLayer)layers.get(l)).weight();
		}
		
		return new NetworkDataPackage(getOutput(), a, z, b, w);
	}
	
	@Override public ModularNetwork clone() {
		ModularNetwork modularNetwork = null;
		
		try {
			modularNetwork = (ModularNetwork) super.clone();
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
			modularNetwork = new ModularNetwork(this.layers);
		}
		
		return modularNetwork;
	}
}
