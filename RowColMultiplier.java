package mat;

public class RowColMultiplier implements Runnable {

	private double[][] M0, M1, MR;
	private int row, col;

	public RowColMultiplier(double[][] M0, double[][] M1, double[][] MR, int row, int col) {
		this.M0 = M0;
		this.M1 = M1;
		this.MR = MR;
		this.row = row;
		this.col = col;
	}

	@Override
	public void run() {
		int len = M1.length;
		double sum = 0;
		for (int i = 0; i < len; i++) {
			sum += M0[row][i] * M1[i][col];
		}
		MR[row][col] = sum;
	}

}
