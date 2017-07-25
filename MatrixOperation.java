package mat;

import java.util.Random;

public class MatrixOperation {

	public double[][] paraMatMul(double[][] M0, double[][] M1) {
		int nRow0 = M0.length;
		int nCol0 = M0[0].length;
		int nRow1 = M1.length;
		int nCol1 = M1[0].length;
		if (nCol0 != nRow1) {
			return null;
		}
		double[][] M = new double[nRow0][nCol1];
		Thread[][] multiplier = new Thread[nRow0][nCol1];
		for (int i = 0; i < nRow0; i++) {
			for (int j = 0; j < nCol1; j++) {
				multiplier[i][j] = new Thread(new RowColMultiplier(M0, M1, M, i, j));
				multiplier[i][j].start();
			}
		}

		for (int i = 0; i < nRow0; i++) {
			for (int j = 0; j < nCol1; j++) {
				try {
					multiplier[i][j].join();
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}

		return M;
	}

	public double[][] matMul(double[][] M0, double[][] M1) {
		int nRow0 = M0.length;
		int nCol0 = M0[0].length;
		int nRow1 = M1.length;
		int nCol1 = M1[0].length;
		if (nCol0 != nRow1) {
			return null;
		}
		double[][] M = new double[nRow0][nCol1];
		for (int i = 0; i < nRow0; i++) {
			for (int j = 0; j < nCol1; j++) {
				for (int k = 0; k < nCol0; k++) {
					M[i][j] += M0[i][k] * M1[k][j];
				}
			}
		}

		return M;
	}

	public void display(double[][] M) {
		int nRow = M.length;
		int nCol = M[0].length;
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				System.out.print(M[i][j] + "\t");
			}
			System.out.println();
		}
	}

	public double[][] genMatrix(int nRow, int nCol) {
		double[][] M = new double[nRow][nCol];
		Random rand = new Random();
		for (int i = 0; i < nRow; i++) {
			for (int j = 0; j < nCol; j++) {
				M[i][j] = rand.nextDouble();
			}
		}
		return M;
	}

}
