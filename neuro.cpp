#include "neuro.h"

//there i use logical sigma 
double sigma(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

// derivative of sigma
double dsigma(double x)
{
	return x * (1.0 - x);
}


// random function (-1,+1)
double rand1()
{
	return (rand() % 32768) / 16384.0 - 1.0;
}

// sqr
double sqr(double x)
{
	return x * x;
}

Neurocolumn::Neurocolumn(int n, int nPrev):
	_size(n),
	_sizePrev(nPrev)
{
	_a.resize(n);
	_b.resize(n);
	_db.resize(n);
	_exp.resize(n);
	_w.resize(n);
	_dw.resize(n);
	for (int i = 0; i < n; i++)
	{
		_a[i] = 0.0;
		_b[i] = rand1();
		_db[i] = 0.0;
		_exp[i] = 0.0;
		_w[i].resize(nPrev);
		_dw[i].resize(nPrev);
		for (int j = 0; j < nPrev; j++)
		{
			_w[i][j] = rand1();
			_dw[i][j] = 0.0;
		}
	}
}

void Neurocolumn::calcActivation(vect& prev)
{
	for (int i = 0; i < _size; i++)
	{
		_a[i] = _b[i];
		for (int j = 0; j < _sizePrev; j++)
		{
			_a[i] += _w[i][j] * prev[j];
		}
	}
	for (int i = 0; i < _size; i++)
	{
		_a[i] = sigma(_a[i]);
	}
}

void Neurocolumn::calcActivation(Neurocolumn* prev)
{
	calcActivation(prev->_a);
}

void Neurocolumn::calcBack(Neurocolumn* prev)
{
	for (int i = 0; i < _sizePrev; i++)
	{
		prev->_exp[i] = 0.0;
	}

	for (int i = 0; i < _size; i++)
	{ //////////////// c is error = sqr(exp - a), a - activation, z = a * w + b - value inside sigma(z)
		double error = 2.0 * (_a[i] - _exp[i]) * dsigma(_a[i]);//// er = dc/da * da/dz   here is hardcoded to logical sigma
		_db[i] += error;
		for (int j = 0; j < _sizePrev; j++)
		{
			_dw[i][j] += error * prev->_a[j]; // dc/dw = dc/da * da/dz * dz/dw = er * a;
			prev->_exp[j] += error * _w[i][j]; // dc/da_prev = dc/da * da/dz * dz/da_ = er * w ; a_ - a from prev
		}
	}
	for (int i = 0; i < _sizePrev; i++)
	{
		prev->_exp[i] = prev->_a[i] - prev->_exp[i];
	}

}

void Neurocolumn::calcBack(vect& prev) // similar ar previous function but need not to calc a_
{
	for (int i = 0; i < _size; i++)
	{ //////////////// c is error = sqr(exp - a), a - activation, z = a * w + b - value inside sigma(z)
		double error = 2.0 * (_a[i] - _exp[i]) * dsigma(_a[i]);//// er = dc/da * da/dz   here is hardcoded to logical sigma
		_db[i] += error;
		for (int j = 0; j < _sizePrev; j++)
		{
			_dw[i][j] += error * prev[j]; // dc/dw = dc/da * da/dz * dz/dw = er * a;
		}
	}
}

void Neurocolumn::correctAll(double c)
{
	for (int i = 0; i < _size; i++)
	{
		_exp[i] = 0.0;
		_a[i] = 0.0;
		_b[i] -= _db[i] * c;
		_db[i] = 0.0;
		for (int j = 0; j < _sizePrev; j++)
		{
			_w[i][j] -= _dw[i][j] * c;
			_dw[i][j] = 0.0;
		}
	}
}

void Neurocolumn::setExpected(vect& expected)
{
	for (int i = 0; i < _size; i++)
	{
		_exp[i] = expected[i];
	}
}

double Neurocolumn::calcError(vect& out)
{
	double error = 0.0;
	for (int i = 0; i < _size; i++)
	{
		error += sqr(out[i] - _a[i]);
	}
	return error / _size;
}

//////////////////////////////////////////////////////////////////////

void Neuronet::init(vector<int>& cols)
{
	column.resize(cols.size() - 1);
	for (int i = 1; i < cols.size(); i++)
	{ // we need n Neurons in this column an in prev (for weights)
		column[i - 1] = new Neurocolumn(cols[i], cols[i - 1]);
	}
}

void Neuronet::calcForward(vect& in)
{
	column.front()->calcActivation(in);
	for (int i = 1; i < column.size(); i++)
	{
		column[i]->calcActivation(column[i - 1]);
	}
}

void Neuronet::calcBack(vect& in)
{
	for (int i = column.size() - 1; i > 0; i--)
	{
		column[i]->calcBack(column[i - 1]);
	}
	column.front()->calcBack(in);
}

void Neuronet::setExpected(vect& out)
{
	column.back()->setExpected(out);
	errorOne = column.back()->calcError(out);
}

void Neuronet::correctAll(double c)
{
	for (auto& it : column)
	{
		it->correctAll(c);
	}
}

void  Neuronet::calcForwardBack(vector<double>& in, vector<double>& out)
{
	calcForward(in); // forward
	setExpected(out);   // get output
	calcBack(in);  // go back
	errorPack += errorOne; // and error
	errorTotal += errorOne; // and error
	errorOne = 0.0;
}

void Neuronet::mixData(vector<vector<double>>& in, vector<vector<double>>& out)
{
	for (int i = 0; i < in.size(); i++) // so it's so terrible... dont look there, please.
	{ // swap two random data
		int n1 = rand() % in.size();
		int n2 = rand() % in.size();
		auto t1 = in[n1];
		in[n1] = in[n2];
		in[n2] = t1;
		auto t2 = out[n1];
		out[n1] = out[n2];
		out[n2] = t2;
	}
}

void Neuronet::learnPack(vector<vector<double>>& in, vector<vector<double>>& out, int n)
{
	errorPack = 0.0; // nulling total error
	errorTotal = 0.0;
	mixData(in, out); // mix vectors
	int iter = 0;
	for (int i = 0; i < in.size(); i++) // learn it n times
	{
		calcForwardBack(in[i], out[i]);
		iter++;
		if (iter == n)
		{
			correctAll(learnRate * errorPack); // and correct weights (w) and offsets (b) as mean rate of all miniset with coef depends on error
			iter = 0;
			errorPack = 0.0;
		}
	}
	if (iter > 0) // if n is not multiple of in.size() we have little tail
	{
		correctAll(learnRate * errorPack);
		errorPack = 0.0;
	}
}
