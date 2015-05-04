#ifndef ConfReader_ConfReader_h
#define ConfReader_ConfReader_h

#include <map>
#include <string>

using std::map;
using std::string;


class ConfReader {
public:
	ConfReader(void);	// hide ctor
	ConfReader(const string conf_file);
	ConfReader(ConfReader const&);	// avoid copy
	void operator=(ConfReader const&);	// avoid assignment

	static ConfReader &getInstance();	// to be used as singleton
    int GetParamValue(const string &key, string &value);

private:
    int readFile(const string &filename);
    int retrieveFilePath(string &filepath);
    int parseContentLine(string &contentLine);
    
    string filePath;
    map<string, string> paramMap;
};


#endif
