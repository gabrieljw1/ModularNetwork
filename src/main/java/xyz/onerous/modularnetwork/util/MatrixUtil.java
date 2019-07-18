package xyz.onerous.modularnetwork.util;

import org.apache.commons.math3.linear.*;

public class MatrixUtil {
	public static double[] hadamard(double[] input1, double[] input2) {
		if (input1.length != input2.length) {
			System.out.println("Error in hadamard product.");
			
			return (double[]) null;
		}
		
		
		double[] hadamardArray = new double[input1.length];
		
		for (int i = 0; i < input1.length; i++) {
			hadamardArray[i] = input1[i] * input2[i];
		}
		
		return hadamardArray;
	}
	
	public static double dot(double[] input1, double[] input2) {
		if (input1.length != input2.length) {
			System.out.println("Error in dot product.");
			
			return 9999999999999999999999.0;
		}
		
		
		double weightedSum = 0;
		
		for (int i = 0; i < input1.length; i++) {
			weightedSum += input1[i] * input2[i];
		}
		
		return weightedSum;
	}
	
	public static double[] multiply(double[][] input1, double[] input2) {
		Array2DRowRealMatrix input1Matrix = new Array2DRowRealMatrix(input1);
		Array2DRowRealMatrix input2Matrix = new Array2DRowRealMatrix(input2);
		
		return input1Matrix.multiply(input2Matrix).transpose().getData()[0];
	}
	
	public static double[] multiplyWithFirstTranspose (double[][] input1, double[] input2) {
		Array2DRowRealMatrix input1Matrix = new Array2DRowRealMatrix(input1);
		Array2DRowRealMatrix input2Matrix = new Array2DRowRealMatrix(input2);
		
		input1Matrix = (Array2DRowRealMatrix)input1Matrix.transpose();
		
		return input1Matrix.multiply(input2Matrix).transpose().getData()[0];
	}
	
	public static double[][] multiplyWithSecondTranspose (double[] input1, double[] input2) {
		Array2DRowRealMatrix input1Matrix = new Array2DRowRealMatrix(input1);
		Array2DRowRealMatrix input2Matrix = new Array2DRowRealMatrix(input2);
		
		input2Matrix = (Array2DRowRealMatrix)input2Matrix.transpose();
		
		return input1Matrix.multiply(input2Matrix).getData();
	}
	
	/**
	 * @param input1
	 * @param input2
	 * @return input1 - input2 for each value position
	 */
	public static double[] subtract(double[] input1, double[] input2) {
		if (input1.length != input2.length) {
			System.out.println("Error in matrix sub.");
			
			return (double[]) null;
		}
		
		
		double[] returnArray = new double[input1.length];
		
		for (int i = 0; i < input1.length; i++) {
			returnArray[i] = input1[i] - input2[i];
		}
		
		return returnArray;
	}
	
	/**
	 * @param input1
	 * @param input2
	 * @return input1 + input2 for each value position
	 */
	public static double[] add(double[] input1, double[] input2) {
		if (input1.length != input2.length) {
			System.out.println("Error in add.");
			
			return (double[]) null;
		}
		
		
		double[] returnArray = new double[input1.length];
		
		for (int i = 0; i < input1.length; i++) {
			returnArray[i] = input1[i] + input2[i];
		}
		
		return returnArray;
	}
	
	public static double[][] add(double[][] input1, double[][] input2) {
		if (input1.length != input2.length || input1[0].length != input2[0].length) {
			System.out.println("Error in add.");
			
			return (double[][]) null;
		}
		
		
		double[][] returnArray = input1.clone();
		
		for (int i = 0; i < input1.length; i++) {
			for (int j = 0; j < input1[i].length; j++) {
				returnArray[i][j] = input1[i][j] + input2[i][j];
			}
		}
		
		return returnArray;
	}
	
	public static double[][][] add(double[][][] input1, double[][][] input2) {
		if (input1.length != input2.length || input1[1].length != input2[1].length || input1[1][0].length != input2[1][0].length) {
			System.out.println("Error in add.");
			
			return (double[][][]) null;
		}
		
		
		double[][][] returnArray = input1.clone();
		
		for (int i = 0; i < input1.length; i++) {
			for (int j = 0; j < input1[i].length; j++) {
				for (int k = 0; k < input1[i][j].length; k++) {
					returnArray[i][j][k] = input1[i][j][k] + input2[i][j][k];
				}
			}
		}
		
		return returnArray;
	}
	
	public static double[] scalarMultiply(double[] matrix, double scale) {		
		double[] returnArray = new double[matrix.length];
		
		for (int i = 0; i < matrix.length; i++) {
			returnArray[i] = matrix[i] * scale;
		}
		
		return returnArray;
	}
	
	public static double[][] scalarMultiply(double[][] matrix, double scale) {		
		double[][] returnArray = matrix.clone();
		
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				returnArray[i][j] = matrix[i][j] * scale;
			}
		}
		
		return returnArray;
	}
}
