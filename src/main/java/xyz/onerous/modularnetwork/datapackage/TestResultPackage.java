package xyz.onerous.modularnetwork.datapackage;

import xyz.onerous.modularnetwork.util.ArrayUtil;
import xyz.onerous.modularnetwork.util.MatrixUtil;

public class TestResultPackage {
	public int    numTests;
	public double percentageCorrect;
	
	public double[]  outputNeuronValues;
	public int[]     outputNeuronIndeces;
	public boolean[] ifCorrect;
	
	public TestResultPackage(int numTests, double percentageCorrect, double[] outputNeuronValues, int[] outputNeuronIndeces, boolean[] ifCorrect) {
		this.numTests = numTests;
		this.percentageCorrect = percentageCorrect;
		
		this.outputNeuronValues = outputNeuronValues;
		this.outputNeuronIndeces = outputNeuronIndeces;
		this.ifCorrect = ifCorrect;	
	}
	
	public String toString() {
		String result = "|  Test  ||  Network Result  ||  Correct?    ||  Confidence\n";
		
		double[] percentageLevels = MatrixUtil.scalarMultiply(outputNeuronValues, 100);
		
		for (int t = 1; t < numTests; t++) {
			String testNumber = t + "";
			String networkResult;
			
			if (ifCorrect[t]) {
				networkResult = "correct  ";
			} else {
				networkResult = "incorrect";
			}
			
			for (int i = (int)(Math.log10(t)+1); i < 3; i++) {
				if (i < 0) { i = 1; }
				
				testNumber += " ";
			}
			
			result += "|  " + testNumber + "   ||       " + outputNeuronIndeces[t] + "          ||  " + networkResult + "   ||   " + Math.round(percentageLevels[t]) + "%\n";
		}
		
		return result + "\n\nTest Results for ID (" + this.hashCode() + ")\n"
				+ "  #Tests:   " + numTests + "\n"
				+ "  %Correct: " + 100.0 * percentageCorrect + "%\n"
				+ "  Average Confidence Level: " + ArrayUtil.mean(outputNeuronValues);
	}
}
