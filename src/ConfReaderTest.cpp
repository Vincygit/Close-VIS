#include <iostream>
#include <string>
using namespace std;
#include "ConfReader.h"
#include <stdio.h>

int amain(int argc, char ** argv)
{
	char buffer[1000];
	string value = string("asdasd").append("asdasd");


	sprintf(buffer, "%sImage_%d.JPG", value.c_str(), 21);
	string output = buffer + string("output.obj");

	value += "asdd";
	cout<<output<<endl;
	ConfReader cr("/home/vincy/vincy.txt");
	cr.GetParamValue("vincy", value);
	std::cout<<"vincy:" << value << "]"<<std::endl;

	cr.GetParamValue("data_folder", value);
	std::cout<<"data_folder:" << value << "]"<<std::endl;

	return 0;
}

