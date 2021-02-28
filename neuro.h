#pragma once

#include <cmath>
#include <vector>

using namespace std;

class Neurocolumn;

class Neuron
{
public:

	vector<double> w;
	vector<double> dw;
	double b;
	double db;
	double er;
	double a;
	double exp;

	void init(int nPrev);
	void correct();
	void correct(double c);

	void calcForward(vector<double>& prev);
	void calcForward(Neurocolumn& prev);
};

class Neurocolumn
{
public:
	vector<Neuron> neu;

	void init(int n, int nPrev);
	void correct();
	void correct(double c);

	void calcForward(vector<double>& prev);
	void calcForward(Neurocolumn& prev);

	void calcBack(vector<double>& prev);
	void calcBack(Neurocolumn& prev);
};


class Neuronet
{
public:
	vector<Neurocolumn> column;
	double learnRate = 0.1;

	void init(vector<int>& cols);
	void setExp(vector<double>& out);
	void correct();
	void correct(double c);

	void calcForward(vector<double>& in);
	void calcBack(vector<double>& in);

	void calcForwardBack(vector<double>& in, vector<double>& out);

	void learn(vector<vector<double>>& in, vector<vector<double>>& out, int n);



};

