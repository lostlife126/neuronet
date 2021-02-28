#include "main.h"

bool loadFisher(vector<vector<double>>& in, vector<vector<double>>& out)
{
	fstream file("iris.data");
	if (!file.is_open())
	{
		cout << "File 'iris.data' is not found!" << endl;
		return false;
	}

	while (!file.eof())
	{
		in.push_back({});
		in.back().resize(4);
		out.push_back({});
		out.back().resize(3);
		file >> in.back()[0];
		file >> in.back()[1];
		file >> in.back()[2];
		file >> in.back()[3];
		out.back()[0] = 0.0;
		out.back()[1] = 0.0;
		out.back()[2] = 0.0;
		string caption;
		file >> caption;
		if (caption.size()==11)//if (caption.c_str() == "Iris-setosa")
		{
			out.back()[0] = 1.0;
		}
		if (caption.size() == 15)//if (caption.c_str() == "Iris-versicolor")
		{
			out.back()[1] = 1.0;
		}
		if (caption.size() == 14)//if (caption.c_str() == "Iris-virginica")
		{
			out.back()[2] = 1.0;
		}
	}
	file.close();

	return true;
}


int main()
{
	vector<int> par{4,4,3};

	Neuronet n;
	n.init(par);


	vector<vector<double>> in;
	vector<vector<double>> out;

	if (!loadFisher(in, out))
	{
		cout << "no data - no deal!"<<endl;
		return 1;
	}

	n.calcForward(in[0]);
	cout << "1 = " << n.column.back().neu[0].a << "  2 = " << n.column.back().neu[1].a << "  3 = " << n.column.back().neu[2].a << endl;

	n.learn(in, out, 50);

	n.calcForward(in[0]);
	cout << "1 = " << n.column.back().neu[0].a << "  2 = " << n.column.back().neu[1].a << "  3 = " << n.column.back().neu[2].a << endl;

	return 0;
}