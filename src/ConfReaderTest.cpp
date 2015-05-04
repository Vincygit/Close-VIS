#include <iostream>

using namespace std;
#include "ConfReader.h"

int main(int argc, char ** argv)
{
	string value;
	ConfReader cr("/home/vincy/vincy.txt");
	cr.GetParamValue("vincy", value);
	std::cout<<"vincy:" << value << "]"<<std::endl;

	cr.GetParamValue("data_folder", value);
	std::cout<<"data_folder:" << value << "]"<<std::endl;

	return 0;
}

