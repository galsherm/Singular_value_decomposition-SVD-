/*
	This is project of matrix decomposition with Singular value decomposition (SVD) method,
	Created by Nadav Shwarz and Gal Sherman on August 2021.
*/

#include<iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <math.h>
#include<vector>
#include<tuple>
#include <functional>
#include<chrono>		//for timer operation
#include <cstdio>		//for remove file operation
#include <random>		//for generate random numbers


//Constants
#define PI 3.14159265358979323846	// Pi
#define TOTAL 1.0e-9	//Total

using namespace std;
using time_type = std::chrono::time_point<std::chrono::steady_clock>;


//Function Declaration:

//Methods for Matrix Operation:
double vec_length(vector<double>& v);

void transpose(vector<vector<double>>& matrix);

vector<vector<double>> dot_product
(vector<vector<double>>& matrix1, vector<vector<double>> matrix2, 
	std::function<void(double&)> F = nullptr);

double dot_product(vector<double>& v1, vector<double>& v2, 
	std::function<void(double&)> F = nullptr);

vector<double> dot_product(double scalar, vector<double>& v);

vector<vector<double>> rotation_product
(vector<vector<double>>& matrix, vector<vector<double>> rotation
	, int p, int q);


vector<vector<double>> diagonal_multiplication(vector<vector<double>>& matrix,
	vector<vector<double>>& diagonal);

vector<vector<double>> eye_matrix(int n);

vector<double> get_diag(vector<vector<double>>& matrix);

vector<vector<double>> diag_to_matrix(vector<double>& diagonal, int size);

vector<double> get_column(vector<vector<double>>& matrix, int column);


//Methods for print vectors/matrix and for reading input file:


void print_vec(vector<double>& v);

void print_matrix(vector<vector<double>>& matrix);

vector<vector<double>> read_file(std::string file_name);

vector<double> read_line(std::string line);

bool write_to_file(std::string file_name, vector<vector<double>>& mtx);

vector<vector<double>> generate_matrix(int size);


//Methods for Jacobi Eigenvalue Algorithm:
tuple<double, int, int> find_max_num(vector<vector<double>>& matrix);

tuple<vector<vector<double>>, double, double> calce_J_matrix(
	vector<vector<double>>& matrix, int p, int q);

void calc_matrix(vector<vector<double>>& mtx, 
	double cos, double sin, int i, int j);

bool check_and_update(vector<vector<double>>& matrix);

std::tuple<vector<vector<double>>, vector<double>, int> 
Jacobi(vector<vector<double>> matrix);

vector<std::tuple<double, vector<double>>>
rearrange(vector<vector<double>>& eigenvectors, vector<double>& lamdas);

std::tuple<vector<vector<double>>, vector<double>, vector<vector<double>>>
SVD(vector<vector<double>>& input_matrix);

void check_decomposition(vector<vector<double>>& input_matrix,
vector<vector<double>> U, vector<double> Sigma, vector<vector<double>> V_T);


//Methods for timer:
namespace my_timer {
	time_type start_timer();
	void end_timer(time_type start);
};

using namespace my_timer;


int main() {
	
	vector<vector<double>> matrix;
	/*matrix = { {1,2}, {2,4} };*/

	/*double t = sqrt(2);
	matrix = {{1,t,2},
		      {t,3,t},
			  {2,t,1} }; */

	/*matrix = { {1,2,7},{2,5,1},{7,1,6} }; /**/

	/*matrix = {{4,-30,60,-35},
				  {-30,300,-675,420},
				  {60,-675,1620,-1050},
				  {-35,420,-1050,700} };/**/


	matrix = generate_matrix(6);
	write_to_file("Input_matrix.txt", matrix);

	matrix = read_file("Input_matrix.txt");

	//Start counting the time.
	time_type start = start_timer();

	std::cout << "Input matrix:" << std::endl;
	print_matrix(matrix);


	auto tuple1 = SVD(matrix);
	vector<vector<double>> U = std::get<0>(tuple1);
	vector<double> Sigma = std::get<1>(tuple1);
	vector<vector<double>> V_T = std::get<2>(tuple1);

	std::cout << "\nSVD:\n\n";

	std::cout << "U = " << std::endl;
	print_matrix(U);
	std::cout << "S = " << std::endl;
	print_vec(Sigma);
	std::cout << "V.T = " << std::endl;
	print_matrix(V_T);

	check_decomposition(matrix, U, Sigma, V_T);

	//Stop the time counting, and print report.
	end_timer(start);
	
	return 0;
}


////Methods for time:

namespace my_timer {
	time_type start_timer() {
		return chrono::steady_clock::now();
	}

	void end_timer(time_type start) {
		//Stop count the time.
		auto end = chrono::steady_clock::now();

		// Store the time difference between start and end
		auto diff = end - start;
		cout << "\nTotal Time Taken = \n";
		cout << chrono::duration <double, milli>(diff).count() << " ms" << endl;
	}
};


//Matrix Operation Methods:


/*
	This method return the vetor length.
*/
double vec_length(vector<double>& v) {
	double sum = 0;
	for (auto& num : v) {
		sum += pow(num, 2);
	}

	return sqrt(sum);
}


/*
This method performe the matrix transpose operation.
Before transpose operation:
{ {1.2,3.4,4},
  {3,4,5.5},
  {2.2, 5, 6}
 }

 After transpose operation:
 { {1.2, 3, 2.2},
   {3.4, 4, 5},
   {4, 5.5, 6}
 }
*/
void transpose(vector<vector<double>>& matrix) {
	
	int row = matrix.size();
	int col = matrix[0].size();

	//pre-allocation of the vector size: 
	vector<double> v (row);
	vector<vector<double>> new_matrix (col, v);

	for (int i = 0; i < col; i++)
	{
		for (int j = 0; j < row; j++)
		{
			new_matrix[i][j] = matrix[j][i];
		}
	}

	matrix = new_matrix;
}


/*
	dot_product method perform matrix multiplication, and return the result.
*/
vector<vector<double>> dot_product(vector<vector<double>>& matrix1,
	vector<vector<double>> matrix2, std::function<void(double&)> F) {

	int row_matrix1 = matrix1.size();
	int col_matrix1 = matrix1[0].size();
	int row_matrix2 = matrix2.size();
	int col_matrix2 = matrix2[0].size();

	//The case that  matrix multiplication isn't define.
	if (col_matrix1 != row_matrix2) {
		std::string str = "Cann't multiply (" +
			std::to_string(row_matrix1) + ", " + std::to_string(col_matrix1) +
			") by (" + std::to_string(row_matrix2) + ", " + std::to_string(col_matrix2) + ").";
		std::cout << str << std::endl;
		throw std::invalid_argument(str);
	}

	//performe transpose to matrix2 
	transpose(matrix2);
	int row_matrix2_T = matrix2.size();

	//pre-allocation of the vector size:
	vector<double> v(row_matrix2_T);
	vector<vector<double>> new_matrix(row_matrix1, v);

	for (int i = 0; i < row_matrix1; i++)
	{
		for (int j = 0; j < row_matrix2_T; j++)
		{
			double num = dot_product(matrix1[i], matrix2[j], F);
			new_matrix[i][j] = num;
		}
	}

	return new_matrix;
}



/*
	dot_product method return the result of the multiplication between vector1 and vector2.
*/
double dot_product(vector<double>& v1, vector<double>& v2, 
	std::function<void(double&)> F) {
	int n;

	if ((n = v1.size()) != v2.size()) {
		std::string str = "vectors are not in the same size.";
		throw std::invalid_argument(str);
	}

	double sum = 0;
	for (int i = 0; i < n; i++)
	{
		sum += (v1[i] * v2[i]);
	}

	if (F != nullptr)
		F(sum);

	return sum;
}


/*
	dot_product method return the result of the multiplication between given scalar and vector.
*/
vector<double> dot_product(double scalar, vector<double>& v) {

	for (int i = 0; i < v.size(); i++)
	{
		v[i] = scalar * v[i];
	}
	return v;
}


/*
	rotation_product method execute matrix multiplication of regular matrix and rotation matrix.
*/
vector<vector<double>> rotation_product(vector<vector<double>>& matrix,
	vector<vector<double>> rotation, int p, int q) {
	int row_matrix = matrix.size();
	int col_matrix = matrix[0].size();
	int row_rotation = rotation.size();
	int col_rotation = rotation[0].size();


	//The case that  matrix multiplication isn't define.
	if (col_matrix != row_rotation) {
		std::string str = "Cann't multiply (" +
			std::to_string(row_matrix) + ", " + std::to_string(col_matrix) +
			") by (" + std::to_string(row_rotation) + ", " + std::to_string(col_rotation) + ").";
		throw std::invalid_argument(str);
	}

	//performe transpose to matrix2 
	transpose(rotation);

	//changes is only have to apply in 2 matrix's columns only - columns p and q.
	const int inner_iteration = 2;

	for (int i = 0; i < row_matrix; i++)
	{
		double index_ip;
		double index_iq;
		for (int j = 0; j < inner_iteration; j++)
		{
			if (j == 0)
				index_ip = dot_product(matrix[i], rotation[p]);
			else
				index_iq = dot_product(matrix[i], rotation[q]);
		}
		matrix[i][p] = index_ip;
		matrix[i][q] = index_iq;
	}
	return matrix;
}


/*
	diagonal_multiplication method execute matrix multiplication of regular matrix
	and diagonal matrix.
*/
vector<vector<double>> diagonal_multiplication(vector<vector<double>>& matrix,
	vector<vector<double>>& diagonal) {
	int row_matrix = matrix.size();
	int col_matrix = matrix[0].size();
	int row_diagonal = diagonal.size();
	int col_diagonal = diagonal[0].size();


	//The case that  matrix multiplication isn't define.
	if (col_matrix != row_diagonal) {
		std::string str = "Cann't multiply (" +
			std::to_string(row_matrix) + ", " + std::to_string(col_matrix) +
			") by (" + std::to_string(row_diagonal) + ", " + std::to_string(col_diagonal) + ").";
		throw std::invalid_argument(str);
	}

	for (int i = 0; i < row_matrix; i++)
	{
		for (int j = 0; j < row_diagonal; j++)
			matrix[i][j] = matrix[i][j] * diagonal[j][j];
	}
	return matrix;
}




/*
	eye_matrix method return the identity matrix of size n.
*/
vector<vector<double>> eye_matrix(int n) {

	//pre-allocation of the vector size:
	vector<double> v (n, 0);
	vector<vector<double>> I(n,v);

	for (int i = 0; i < n; i++)
		I[i][i] = 1;

	return I;
}


/*
	get_diag method return vector that contains the elements that on the matrix diagonal.
*/
vector<double> get_diag(vector<vector<double>>& matrix) {

	int row = matrix.size();
	int col = matrix[0].size();

	//Check that the given matrix is square matrix
	if (row != col) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	//pre-allocation of the vector size: 
	vector<double> diagonal (row);

	for (int i = 0; i < row; i++)
	{
		diagonal[i] = matrix[i][i];
	}
	return diagonal;
}


/*
	diag_to_matrix method convert a vector to diagonal matrix in size size_val value.
*/
vector<vector<double>> diag_to_matrix(vector<double>& diagonal, int size_val) {

	int n = diagonal.size();

	//The case that the desire matrix size is smaller than 
	// the amount of elements in diagonal vectors.
	if (n > size_val) {
		std::string str = "The desire matrix size is smaller than the amount of elements in diagonal vectors";
		throw std::invalid_argument(str);
	}

	//pre-allocation of the vector size:
	vector<double> v(size_val, 0);
	vector<vector<double>> new_matrix(size_val, v);

	for (int i = 0; i < n; i++)
	{
		new_matrix[i][i] = diagonal[i];
	}

	return new_matrix;
}


/*
	get_column method get matrix and column. The method rturn vector 
	with the elements of this matrix's column.
*/
vector<double> get_column(vector<vector<double>>& matrix, int column) {

	//Check that the given column number is not out of range.
	if (matrix[0].size() < column) {
		std::string str = "The given matrix do not have column number " + std::to_string(column) + ".";
		throw std::invalid_argument(str);
	}
	
	int n = matrix.size();
	//pre-allocation of the vector size: 
	vector<double> v(n);

	for (int i = 0; i < n; i++)
	{
		v[i] = matrix[i][column];
	}
	return v;
}



//Method for print vectors/matrix and for reading input file:


/*
	print_vec method print the given vector to the console.
*/
void print_vec(vector<double>& v) {
	bool flag = false;

	for (int i = 0; i < v.size(); i++)
	{
		if (v.size() == 1)
			std::cout << "[" << v[i] << "]" << std::endl;
		else if (i == 0)
			std::cout << "[" << v[i] << ",	";
		else if (i + 1 == v.size())
			std::cout << v[i] << "]" << std::endl;
		else {
			std::cout << v[i] << ",	";

			//The case the given vector is bigger then 10 elements:
			if (i > 2 and v.size() > 10 and not flag) {
				std::cout << "...	";
				i = v.size() - 4;
				flag = true;
			}
		}
	}
}


/*
	print_matrix method print the given matrix to the console.
*/
void print_matrix(vector<vector<double>>& matrix) {
	bool flag = false;

	for (int i = 0; i < matrix.size(); i++)
	{
		if (matrix.size() == 1) {
			std::cout << "[\n ";
			print_vec(matrix[i]);
			std::cout << "]" << std::endl;
		}
		else if (i == 0) {
			std::cout << "[ ";
			print_vec(matrix[i]);
		}
		else if (i + 1 == matrix.size()) {
			std::cout << " ";
			print_vec(matrix[i]);
			std::cout << "] " << std::endl;
		}
		else {
			std::cout << " ";
			print_vec(matrix[i]);

			//The case the given matrix is bigger then 10 elements:
			if (i > 2 and matrix.size() > 10 and not flag) {
				std::cout << " ...	\n";
				i = matrix.size() - 4;
				flag = true;
			}
		}
	}
}


/*
	read_file method read the all content from the input file.
	The method return matrix with all the values from the input file.
*/
vector<vector<double>> read_file(std::string file_name) {
	std::string line;
	fstream newfile;
	vector<vector<double>> matrix;

	
	//ios::in - is mode that represent the operation of open for input operations.
	newfile.open(file_name, ios::in); //open a file to perform read operation using file object

	if (newfile.is_open()) {   //checking that the file is open

		while (getline(newfile, line)) { //read data from file object and put it into string.
			auto vector = read_line(line);
			matrix.push_back(vector);
		}

		newfile.close(); //close the file object.
	}
	return matrix;
}

/*
	read_line method get string value and convert it to vector of double.
*/
vector<double> read_line(std::string line) {

	vector<double> v;

	size_t pos = 0;
	std::string delimiter = ",";

	while ((pos = line.find(delimiter)) != std::string::npos) {
		std::string token = line.substr(0, pos);

		double num = std::stod((const std::string&)token);
		v.push_back(num);

		line.erase(0, pos + delimiter.length());
	}

	//Last number in the line do not have comma after it.
	double num = std::stod((const std::string&)line);
	v.push_back(num);

	return v;
}


/*
	generate_matrix method randomly generate matrix.
*/
vector<vector<double>> generate_matrix(int size) {

	//Initialize parameters:
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<int> dist(0, 80);


	vector<double> v(size);
	vector<vector<double>> new_matrix(size, v);

	for (int i = 0; i < size; i++)
	{
		for (int j = i; j < size; j++)
		{
			new_matrix[i][j] = new_matrix[j][i] = dist(mt);
		}
	}
	return new_matrix;
}


/*
	write_to_file method write the content of the given matrix to the file.
	The method True if succeeded.
*/
bool write_to_file(std::string file_name, vector<vector<double>>& mtx){

	int row = mtx.size();
	int col = mtx[0].size();

	//remove the file if exists - string.c_str() convert string to char*
	std::remove(file_name.c_str());

	ofstream myfile;
	myfile.open(file_name, ios::out | ios::app | ios::binary);

	if (myfile.is_open()) {		//ok, proceed with output

		std::string delimiter = ",";
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				if (not(j + 1 < col))		//Case of the last number in the matrix
					myfile << mtx[i][j];
				else
					myfile << mtx[i][j] << ",";
			}
			if (i + 1 < row)
				myfile << "\n";
		}
		myfile.close(); //close the file object.
		return true;
	}
	return false;
}


//Methods for Jacobi Eigenvalue Algorithm:

/*
	find_max_num method find the largest absolute value off-diagonal element, 
	and return his indexes and value.
	The given matrix is symmetric, so only need to search in the upper triangular matrix.
*/
std::tuple<double, int, int> find_max_num(vector<vector<double>>& matrix) {
	int row = matrix.size();
	int col = matrix[0].size();
	int p, q;
	double max_val;

	for (int i = 0; i < row; i++)
	{
		for (int j = i+1; j < col; j++)
		{
			if (i == 0 and j == 1) {
				max_val = std::abs(matrix[i][j]);
				p = i;
				q = j;
			}
			else if (max_val < std::abs(matrix[i][j])) {
				max_val = std::abs(matrix[i][j]);
				p = i;
				q = j;
			}
		}
	}

	return std::make_tuple(max_val, p, q);
}


/*
	calce_J_matrix method find the rotation matrix that called J:
*/
std::tuple<vector<vector<double>>, double, double>
calce_J_matrix(vector<vector<double>>& matrix, int p, int q) {

	//Check that the given matrix is square matrix
	if (matrix.size() != matrix[0].size()) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	//alocation new identity matrix:
	int n = matrix.size();
	auto J = eye_matrix(n);

	double theta;

	//calculate theta:
	if (matrix[q][q] == matrix[p][p])
		theta = PI / 4;
	else {
		double a = (2 * matrix[p][q]) / (matrix[q][q] - matrix[p][p]);
		theta = 0.5 * atan(a);
	}

	double cosinus, sinus;
	//insert new values to different places in the matrix J :
	J[p][p] = J[q][q] = cosinus = cos(theta);
	J[q][p] = sinus = sin(theta);
	J[p][q] = -1 * sin(theta);


	return std::make_tuple(J, cosinus, sinus);
}


/*
	my_round check if num is smaller than TOTAL value. If so, then num become zero.
*/
void my_round(double& num) {
	if (std::abs(num) < TOTAL)
		num = 0;
}


/*
	For make the performention of Jacobi Eigenvalue Algorithm better,
	matrix multiplication of J.T*A*J is replaced by some elementary operations.
	And then, the all function is O(n) instead of O(n^2).
*/
void calc_matrix(vector<vector<double>>& mtx, double cos, double sin, int i, int j) {
	double a_ii = mtx[i][i];
	double a_ij = mtx[i][j];
	double a_jj = mtx[j][j];
	double a_ji = mtx[j][i];

	mtx[i][i] = pow(cos, 2) * a_ii - 2 * sin * cos * a_ij + pow(sin, 2) * a_jj;
	my_round(mtx[i][i]);

	mtx[j][j] = pow(sin, 2) * a_ii + 2 * sin * cos * a_ij + pow(cos, 2) * a_jj;
	my_round(mtx[j][j]);

	mtx[i][j] = mtx[j][i] = (pow(cos, 2) - pow(sin, 2)) * a_ij + sin * cos * (a_ii - a_jj);
	my_round(mtx[i][j]);
	my_round(mtx[j][i]);

	for (int k = 0; k < mtx.size(); k++)
	{
		if (k != i and k != j) {
			double a_ik = mtx[i][k];
			double a_jk = mtx[j][k];
			mtx[i][k] = mtx[k][i] = cos * a_ik - sin * a_jk;
			mtx[j][k] = mtx[k][j] = sin * a_ik + cos * a_jk;

			my_round(mtx[i][k]); my_round(mtx[k][i]);
			my_round(mtx[j][k]); my_round(mtx[k][j]);
		}
	}
}


/*
	check_and_update method is doing the following things:
	Convert matrix elemnt that is smaller than TOTAL (almost equal to 0) to be 0.
	Check if the given matrix is diagonal.
*/
bool check_and_update(vector<vector<double>>& matrix) {
	//bool isDiagonalMatrix = true;
	int row = matrix.size();
	int col = matrix[0].size();

	//Check that the given matrix is square matrix
	if (row != col) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	for (int i = 0; i < row; i++)
	{
		for (int j = i+1; j < col; j++)
		{
			if (abs(matrix[i][j]) < TOTAL)
				matrix[i][j] = matrix[j][i] = 0;
			else
				return false;
		}
	}
	return true;
}


/*
	Jacobi method Implemntion
	Jacobi Eigenvalue Algorithm: https://en.wikipedia.org/wiki/Jacobi_eigenvalue_algorithm
	The method return Eigenvalues and Eigenvectors.
*/

std::tuple<vector<vector<double>>, vector<double>, int>
Jacobi(vector<vector<double>> matrix) {

	//Check that the given matrix is square matrix
	if (matrix.size() != matrix[0].size()) {
		std::string str = "The given matrix is not square matrix.";
		throw std::invalid_argument(str);
	}

	//Initialize of the variables:
	int n = matrix.size();
	vector<vector<double>> J = eye_matrix(n);

	//Set limit on the number of iterations:
	int max_iterations = 100;
	int cur_iteration_num = 0;

	for (int i = 0; i < max_iterations; i++)
	{
		//Get matrix max number and his index:
		auto tuple1 = find_max_num(matrix);
		double max_val = std::get<0>(tuple1);
		int p = std::get<1>(tuple1);
		int q = std::get<2>(tuple1);

		if (max_val < TOTAL)
			return std::make_tuple(J, get_diag(matrix), cur_iteration_num);

		//Get rotation matrix and get cos and sin values:
		auto tuple2 = calce_J_matrix(matrix, p, q);
		vector<vector<double>> J1 = std::get<0>(tuple2);
		double cosinus = std::get<1>(tuple2);
		double sinus = std::get<2>(tuple2);

		//Calculate the new matrix:
		calc_matrix(matrix, cosinus, sinus, p, q);

		//Calculate the eigenvectors:
		J = rotation_product(J, J1, p, q);		

		cur_iteration_num++;
		if (check_and_update(matrix))
			break;
	}

	return std::make_tuple(J, get_diag(matrix), cur_iteration_num);
}


/*
	rearrange method remove negative and zero Eigenvalues and their Eigenvectors.
	The method return Sorted list, of Eigenvalues and their Eigenvectors.
*/
vector<std::tuple<double, vector<double>>> 
rearrange(vector<vector<double>>& eigenvectors, vector<double>& lamdas) {

	//Initialize of the variables:
	vector<std::tuple<double, vector<double>>> t_vecs;
	bool flag = false;

	for (int i = 0; i < lamdas.size(); i++)
	{
		if (lamdas[i] > 0) {

			auto tuple = std::make_tuple(lamdas[i], get_column(eigenvectors, i));
			if (t_vecs.size() == 0)
				t_vecs.push_back(tuple);
			else {
				for (int j = 0; j < t_vecs.size(); j++)
				{
					auto tuple1 = t_vecs[j];
					if (std::get<0>(tuple1) <= lamdas[i]) {

						// Create Iterator pointing to the desire place:
						auto itPos = t_vecs.begin() + j;

						// Insert element to the desire position in vector:
						t_vecs.insert(itPos, tuple);
						flag = true;
						break;
					}
				}
				if (not flag)
					t_vecs.push_back(tuple);

				//reinsialize
				flag = false;
			}
		}
	}

	return t_vecs;
}


/*
	This method get input matrix and perfome Singular Value Decomposition (SVD).
*/
std::tuple<vector<vector<double>>, vector<double>, vector<vector<double>>> 
SVD(vector<vector<double>>& input_matrix) {
	
	//copy constructor
	vector<vector<double>> AT = input_matrix;
	transpose(AT);

	auto AT_T = dot_product(AT, input_matrix);

	auto tuple1 = Jacobi(AT_T);
	vector<vector<double>> eigenvectors = std::get<0>(tuple1);
	vector<double> eigenvalues = std::get<1>(tuple1);


	std::cout << "Eigenvectors = " << std::endl;
	print_matrix(eigenvectors);
	std::cout << "Eigenvalues = " << std::endl;
	print_vec(eigenvalues);

	auto vec = rearrange(eigenvectors, eigenvalues);

	//Build Sigma matrix - contain the Singular Values in descending order on the main diagonal :

	vector<double> Sigma(vec.size());

	//Build U matrix - (1 / Singular Values) * A * V

	vector<vector<double>> U;

	//Build V.T matrix - contain the transpose of the eigenvectors.

	vector<vector<double>> V_T;

	for (int i = 0; i < vec.size(); i++)
	{
		auto tuple2 = vec[i];

		double lamda = std::get<0>(tuple2);
		vector<double> v1 = std::get<1>(tuple2);

		double s = sqrt(lamda);

		//Create v vector by normalize v1 vectore:
		double vec_norm = vec_length(v1);
		auto v = dot_product(1/vec_norm, v1);
		
		//Create u vector by multiply scalar on dot product of input matrix with v vector ==> (1/s)*input_matix*v:
		double scalar = 1 / s;
		vector<vector<double>> v_t = { v };
		transpose(v_t);

		v_t = dot_product(input_matrix, v_t);
		transpose(v_t);

		auto u = dot_product(scalar, v_t[0]);
	
		//Insert values to the finale data structures:
		Sigma[i] = s;

		U.push_back({ u });

		V_T.push_back({ v });
	}

	transpose(U);
	//transpose(V_T);

	return std::make_tuple(U, Sigma, V_T);
}


/*
	Checking that the U * S * VT is equal to input matrix.
*/
void check_decomposition(vector<vector<double>>& input_matrix,
	vector<vector<double>> U, vector<double> Sigma, vector<vector<double>> V_T) {

	std::cout << "Examination Of The Decomposition: " << std::endl;
	std::cout << "Inpute Matrix = " << std::endl;
	print_matrix(input_matrix);

	std::cout << std::endl;

	std::cout << "Result Of The 3 Matrix Multiplication From SVD Is:\nU*S*V.t =" << std::endl;

	auto U_Sigma = diag_to_matrix(Sigma, U[0].size());

	auto temp_mtx = diagonal_multiplication(U, U_Sigma);

	auto res = dot_product(temp_mtx, V_T, my_round);

	print_matrix(res);
}