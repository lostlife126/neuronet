#include "neuro.h"

double sigma(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double dsigma(double x)
{
	return x * (1.0 - x);
}

double rand1()
{
	return (rand() % 32768) / 16384.0 - 1.0;
}

double sqr(double x)
{
	return x * x;
}

///////////////////////////////////////////////////////////////////////////////

void Neuron::init(int nPrev)
{
	w.resize(nPrev);
	dw.resize(nPrev, 0.0);
	db = 0.0;
	exp = 0.0;
	a = 0.0;
	er = 0.0;
	b = rand1();
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = rand1();
	}
}

void Neuron::correct()
{

	a = 0.0;
	er = 0.0;
	b -= db;
	db = 0.0;
	exp = 0.0;
	for (int i = 0; i < w.size(); i++)
	{
		w[i] -= dw[i];
		dw[i] = 0.0;
	}
}

void Neuron::correct(double c)
{

	a = 0.0;
	er = 0.0;
	b -= db * c;
	db = 0.0;
	exp = 0.0;
	for (int i = 0; i < w.size(); i++)
	{
		w[i] -= dw[i] * c;
		dw[i] = 0.0;
	}
}

void Neuron::calcForward(vector<double>& prev)
{
	a = b;
	for (int i = 0; i < w.size(); i++)
	{
		a += prev[i] * w[i];
	}
	a = sigma(a);
}

void Neuron::calcForward(Neurocolumn& prev)
{
	a = b;
	for (int i = 0; i < w.size(); i++)
	{
		a += prev.neu[i].a * w[i];
	}
	a = sigma(a);
}

///////////////////////////////////////////////////////////////////////////

void Neurocolumn::init(int n, int nPrev)
{
	neu.resize(n);
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].init(nPrev);
	}
}

void Neurocolumn::calcForward(vector<double>& prev)
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].calcForward(prev);
	}
}

void Neurocolumn::calcForward(Neurocolumn& prev)
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].calcForward(prev);
	}
}

void Neurocolumn::calcBack(Neurocolumn& prev)
{
	for (int i = 0; i < prev.neu.size(); i++)
	{
		prev.neu[i].exp = 0.0;
	}

	for (int i = 0; i < neu.size(); i++)
	{
		double er = (neu[i].a - neu[i].exp) * dsigma(neu[i].a);///////////////// here is hardcoded to exp sigma
		neu[i].db += er;
		for (int j = 0; j < neu[i].w.size(); j++)
		{
			neu[i].dw[j] += er * prev.neu[j].a;
			prev.neu[j].exp += er * neu[i].w[j];
		}
	}
}

void Neurocolumn::calcBack(vector<double>& prev)
{
	for (int i = 0; i < neu.size(); i++)
	{
		double er = (neu[i].a - neu[i].exp) * dsigma(neu[i].a);///////////////// here is hardcoded to exp sigma
		neu[i].db += er;
		for (int j = 0; j < neu[i].w.size(); j++)
		{
			neu[i].dw[j] += er * prev[j];
		}
	}
}

void Neurocolumn::correct()
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].correct();
	}
}

void Neurocolumn::correct(double c)
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].correct(c);
	}
}

//////////////////////////////////////////////////////////////////////

void Neuronet::init(vector<int>& cols)
{
	column.resize(cols.size() - 1);
	for (int i = 1; i < cols.size(); i++)
	{
		column[i - 1].init(cols[i], cols[i - 1]);
	}
}

void Neuronet::calcForward(vector<double>& in)
{
	column.front().calcForward(in);
	for (int i = 1; i < column.size(); i++)
	{
		column[i].calcForward(column[i - 1]);
	}
}

void Neuronet::calcBack(vector<double>& in)
{
	for (int i = column.size() - 1; i > 0; i--)
	{
		column[i].calcBack(column[i - 1]);
	}
	column.front().calcBack(in);
}

void Neuronet::setExp(vector<double>& out)
{
	for (int i = 0; i < out.size(); i++)
	{
		column.back().neu[i].exp = out[i];
	}
}

void Neuronet::correct()
{
	for (int i = 0; i < column.size(); i++)
	{
		column[i].correct();
	}
}

void Neuronet::correct(double c)
{
	for (int i = 0; i < column.size(); i++)
	{
		column[i].correct(c);
	}
}

void  Neuronet::calcForwardBack(vector<double>& in, vector<double>& out)
{
	calcForward(in);
	setExp(out);
	calcBack(in);
}

void Neuronet::learn(vector<vector<double>>& in, vector<vector<double>>& out, int n)
{
	int iter = 0;
	for (int i = 0; i < in.size(); i++)
	{
		calcForwardBack(in[i], out[i]);
		iter++;
		if (iter == n)
		{
			correct(1.0 / iter);
			iter = 0;
		}
	}
	if (iter > 0)
	{
		correct(1.0 / iter);
	}
}
