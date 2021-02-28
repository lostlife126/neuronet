#pragma once

#include <cmath>
#include <vector>

using namespace std;

class Neurocolumn;

// Neuron is class stored weights with neurons other layer and offset for calc sigma(a * w + b)
// and corrections of theese parameters
class Neuron
{
public:

	vector<double> w; // weights with prev layer
	vector<double> dw; // corrections of w
	double b; // offset of z = w * a + b
	double db;// correction of b
	double a; // activation
	double exp; // expected activation for back propagation calc

	// nPrev - is count of neurons in prev layer for creating weights
	void init(int nPrev); // initialization (nulling values and creation of vectors)

	// c is factor for correction
	void correct(double c); // correction of weights and offset

	// prev is activations of prev layer (for first after input layer)
	void calcForward(vector<double>& prev); // calc a = sigma(w * a_ + b) for first (next to input layer)

	// prev is prev layer (for internal layer)
	void calcForward(Neurocolumn& prev); // calc a for other layers
};

// layer of neurons. store neurons and can calcForward and back
class Neurocolumn
{
public:
	vector<Neuron> neu; // vector of neurons

	// n is count of neurons in this layer and nPrev - in previous layer
	void init(int n, int nPrev); // initialization

	// c is factor of correction
	void correct(double c); // correct neurons

	// prev - is activation of prev layer
	void calcForward(vector<double>& prev); // forvard calculations for first layer
	void calcForward(Neurocolumn& prev); // and other layer

	void calcBack(vector<double>& prev); // back propagation for last (first of layers)
	void calcBack(Neurocolumn& prev); // back propagation
};


class Neuronet
{
public:
	vector<Neurocolumn> column; // layers of neurons (input is not here)
	double learnRate = 1.0; // learn rate for changing speed of learning
	double total_error; // store sum of error in all learning set
	double error;        // store error in one step forward-back

	// cols is vector of neurons in all layers vector {3,4,4,4,5} - 3 input neurons, 4x3 internal layers, 5 output neurons
	void init(vector<int>& cols); 

	// out is output data for test
	void setExp(vector<double>& out); // put expected values in last layer for back prop and error

	// c is coef that factor for correction
	void correct(double c); // correction w and b after miniset of forward-back steps 

	// in - is input data for test
	void calcForward(vector<double>& in); // calculation forward and calc answers =)
	// in - is input data for test
	void calcBack(vector<double>& in); // calculation back (expected values must be in last layer)

	// in - is input data for test
	// out is output data for test
	void calcForwardBack(vector<double>& in, vector<double>& out); // one step forward-back for one in and out vector
	
	// vector in is vector of input data
	// vector in is vector of input data 
	void learn(vector<vector<double>>& in, vector<vector<double>>& out, int n); // set of steps using all data (in.size() forward-back steps) with n corrections

	// vector in is vector of input data
	// vector in is vector of input data 
	void mixData(vector<vector<double>>& in, vector<vector<double>>& out); // mixing vectors in and out for more uniform and homogeneous data


};

