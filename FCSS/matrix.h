#pragma once
#include <vector>

#ifndef _MATRIX_H_
#define _MATRIX_H_

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

//Caluculate volume of supervoxel in high dimension
class Matrix{
public:
	void MatrixMultiply(
		std::vector<std::vector<double> >& firstMatrix,
		std::vector<std::vector<double> >& secondMatrix,
		std::vector<std::vector<double> >& mult,
		const int rowFirst,
		const int columnFirst,
		const int columnSecond)
	{
		// Initializing elements of matrix mult to 0.
		for (int i = 0; i < rowFirst; ++i)
			for (int j = 0; j < columnSecond; ++j)
				mult[i][j] = 0;
		// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
		for (int i = 0; i < rowFirst; ++i)
			for (int j = 0; j < columnSecond; ++j)
				for (int k = 0; k < columnFirst; ++k)
					mult[i][j] += firstMatrix[i][k] * secondMatrix[k][j];
	}

	void MatrixMultiplyVector(
		std::vector<std::vector<double> >& firstMatrix,
		std::vector<double>& secondvector,
		std::vector<double>& multvector,
		const int rowFirst,
		const int columnFirst)
	{
		// Initializing elements of matrix mult to 0.
		for (int i = 0; i < rowFirst; ++i)
			multvector[i] = 0;
		// Multiplying matrix firstMatrix and secondMatrix and storing in array mult.
		for (int i = 0; i < rowFirst; ++i)
			for (int k = 0; k < columnFirst; ++k)
				multvector[i] += firstMatrix[i][k] * secondvector[k];
	}

	int Gaussian(std::vector<std::vector<double> >& A, std::vector<double>& b, std::vector<double>& x, const int n)
	{
		for (int i = 0; i < n; i++)
			if (A[i][i] == 0)
			{
				printf("can't use gaussian method\n");
				return -1;
			}
		int i, j, k;
		std::vector<double> c(n, 0);    // Store the coefficients of elementary row transformation
		// Gauss elimination, total n-1 steps
		for (k = 0; k < n - 1; k++)
		{
			// Calculate the coefficients for the k-th elementary row transformation
			for (i = k + 1; i < n; i++)
				c[i] = A[i][k] / A[k][k];

			// The k-th elimination
			for (i = k + 1; i < n; i++)
			{
				for (j = 0; j < n; j++)
				{
					A[i][j] = A[i][j] - c[i] * A[k][j];
				}
				b[i] = b[i] - c[i] * b[k];
			}
		}

		// First calculate the last unknown
		x[n - 1] = b[n - 1] / A[n - 1][n - 1];
		// Then calculate remaining unknowns
		for (i = n - 2; i >= 0; i--)
		{
			double sum = 0;
			for (j = i + 1; j < n; j++)
			{
				sum += A[i][j] * x[j];
			}
			x[i] = (b[i] - sum) / A[i][i];
		}
		return 0;
	}

	void Transpose(std::vector<std::vector<double> >& A, std::vector<std::vector<double> >& At, const int row, const int col){
		for (int i = 0; i < col; i++)
			for (int j = 0; j < row; j++)
				At[i][j] = A[j][i];
	}

	double DistVec(std::vector<double>& x, std::vector<double>& y, const int n){
		double dist = 0;
		for (int i = 0; i < n; i++)
			dist += (x[i] - y[i]) * (x[i] - y[i]);
		return sqrt(dist);
	}
};

#endif