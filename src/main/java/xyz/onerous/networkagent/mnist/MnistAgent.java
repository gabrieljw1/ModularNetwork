package xyz.onerous.networkagent.mnist;

import java.util.List;
import java.util.Timer;

import xyz.onerous.modularnetwork.ModularNetwork;
import xyz.onerous.modularnetwork.datapackage.TestResultPackage;
import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.learner.multithread.MTBatchTrainer;
import xyz.onerous.modularnetwork.type.ActivationType;
import xyz.onerous.modularnetwork.type.LossType;
import xyz.onerous.modularnetwork.util.ArrayUtil;

public class MnistAgent {
	private List<int[][]> images;
	protected int[] labels;
	private List<int[][]> testImages;
	protected int[] testLabels;
	
	protected double[][] imageData;
	protected double[][] testImageData;
	
	protected ModularNetwork modularNetwork;
	
	protected static final String IMAGES_FILE_PATH = "./src/main/resources/train-images.idx3-ubyte";
	protected static final String LABELS_FILE_PATH = "./src/main/resources/train-labels.idx1-ubyte";
	protected static final String TEST_IMAGES_FILE_PATH = "./src/main/resources/t10k-images.idx3-ubyte";
	protected static final String TEST_LABELS_FILE_PATH = "./src/main/resources/t10k-labels.idx1-ubyte";
	
	protected final double learningRate = 0.001;
	protected final boolean usingSoftmax = true;
	protected final LossType lossType = LossType.CrossEntropy;
	
	public MnistAgent() {
		this.images = MnistReader.getImages(IMAGES_FILE_PATH);
		this.labels = MnistReader.getLabels(LABELS_FILE_PATH);
		this.testImages = MnistReader.getImages(TEST_IMAGES_FILE_PATH);
		this.testLabels = MnistReader.getLabels(TEST_LABELS_FILE_PATH);
		
		imageData = new double[images.size()][];
		testImageData = new double[testImages.size()][];
		
		for (int i = 0; i < imageData.length; i++) {
			imageData[i] = ArrayUtil.standardize( ArrayUtil.flattenArray(images.get(i)) );
		}
		
		for (int i = 0; i < testImageData.length; i++) {
			testImageData[i] = ArrayUtil.standardize( ArrayUtil.flattenArray(testImages.get(i)) );
		}
	}
	
	public void performEpochs(int batchSize, int numberOfEpochs) throws InvalidInputLengthException {
		for (int i = 0; i < numberOfEpochs; i++) {
			modularNetwork.performEpoch(imageData, labels, batchSize, lossType, usingSoftmax, learningRate);
		}
	}
	
	public TestResultPackage performTest(int startIndex, int endIndex) throws InvalidInputLengthException {
		if (startIndex < 0 || endIndex > testImages.size() || startIndex >= endIndex) { return (TestResultPackage) null; }
		
		TestResultPackage testResults = modularNetwork.performTest(ArrayUtil.clipArray(testImageData, startIndex, endIndex), ArrayUtil.clipArray(testLabels, startIndex, endIndex));
		
		return testResults;
	}
	
	public void generateNetwork() {
		int nInput = imageData[0].length;
		int nOutput = 10;
		
		this.modularNetwork = new ModularNetwork(nInput);
		
		modularNetwork.addLayer(ActivationType.ReLU, 300);
		modularNetwork.addLayer(ActivationType.ReLU, 100);
		
		modularNetwork.addLayer(ActivationType.Sigmoid, nOutput);
		
		modularNetwork.initializeWeightsDynamic();
		
		modularNetwork.multithreadingEnable = true;
	}
	
	public static void main(String[] args) {
		MnistAgent mnistAgent = new MnistAgent();
		
		mnistAgent.generateNetwork();
		
		try {
			long startTime = System.nanoTime();
			mnistAgent.performEpochs(4, 5);
			long endTime = System.nanoTime();
			System.out.println(mnistAgent.performTest(0, 500));
			
			System.out.println("Epochs took: " + (endTime - startTime)/1000000000.0 + "s");
		} catch (InvalidInputLengthException e) {
			e.printStackTrace();
		}
	}
}
