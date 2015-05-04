#include <iostream>
#include <fstream>
#include <cctype>
#include <algorithm>

#include "ConfReader.h"

using std::wifstream;
using std::find;
using std::getline;


#define ANNOTATION_SYMBOL1 '#'
#define ANNOTATION_SYMBOL2 ';'

#ifdef WIN32
#define FILE_SEPERATOR "\\"
#else
#define FILE_SEPERATOR "/"
#endif

using namespace std;

static inline string ltrim(string s) {
	int firstC = s.find_first_not_of(' ');
	if(firstC < 0 )
		return string("");
	else {
		string result = s.substr(firstC, s.length() - firstC);
		return result;
	}
}

// trim from end
static inline string rtrim(string s) {
	int lastC = s.find_last_not_of(' ');
	if(lastC < 0 )
		return string("");
	else{
		string result = s.substr(0, lastC+1);
		return result;
	}
}

static inline string uncomment(string s) {
	int lastC = s.find_first_of(ANNOTATION_SYMBOL1);
	int lastC2 = s.find_first_of(ANNOTATION_SYMBOL2);

	if(lastC < 0 && lastC2 < 0)
		return s;

	if(lastC > 0) {
		if(lastC2 > 0)
			lastC = lastC > lastC2 ? lastC2 : lastC;
	} else {
		lastC = lastC2;
	}
	string result = s.substr(0, lastC);
	return rtrim(result);
}
// trim from both ends
static inline string trim(string &s) {
	return rtrim(ltrim(s));
}

ConfReader::ConfReader(void) {
	// default
}

ConfReader::ConfReader(const string conf_file) {
	// default
	this->filePath = string(conf_file);
}

ConfReader::ConfReader(const ConfReader &) {
	// default
}

void ConfReader::operator =(ConfReader const&) {
	// default
}

ConfReader& ConfReader::getInstance() {
	static ConfReader instance;
	return instance;
}


/**
 *
 * get value by given key.
 * @param key			(in)key
 * @param valueBuffer	(out)value
 * @return
 *		-1:error
 *		0:success
 *		1:not found the key
 */
int ConfReader::GetParamValue(const string &key, string &valueBuffer) {
	string filepath;
	if (paramMap.empty()) {
		int rtncode = retrieveFilePath(filepath);
		if(rtncode == 0){
			readFile(filepath);
		} else {
			return -1;
		}
	}

	map<string, string>::iterator iter = paramMap.find(key);
	if (iter != paramMap.end()) {
		valueBuffer = iter->second;
		return 0;
	} else{
		//key not found.
		valueBuffer = string("Not Found");
		return -1;
	}    
}

/**
 *
 * read config file, add <key,value> into map.
 * @param filepath (in)line text
 * @param return
 *		-1:error,invalid line
 *		0:success
 */
int ConfReader::readFile(const string &filename) {
	ifstream infile(filename.c_str());
	string buffer;
	while (getline(infile, buffer)) {
		parseContentLine(buffer);
	}

	return 0;
}

/**
 *
 * handle single line text,then add <key,value> into map.
 * @param filepath (in)line text
 * @param return
 *		-1:error,invalid line
 *		0:success
 */
int ConfReader::parseContentLine(string &contentLine) {
	contentLine = trim(contentLine);

	if (contentLine.size() < 1) {
		return 0;   // blank line
	}

	if (contentLine.at(0) == ANNOTATION_SYMBOL1
			|| contentLine.at(0) == ANNOTATION_SYMBOL2) {
		return 0;   // comment
	}

	contentLine = uncomment(contentLine);

	string::size_type equalPos = contentLine.find_first_of("=");
	string::size_type startPos = 0;
	string::size_type endPos = contentLine.size() - 1;
	if (equalPos <= startPos || equalPos > endPos) {
		return -1; // invalid line
	}

	string key = rtrim(contentLine.substr(startPos, equalPos ));
	string value = ltrim(contentLine.substr(equalPos + 1, endPos));

	paramMap.insert(std::make_pair(key, value));

	return 0;
}

/**
 *
 * get config file absolute path.
 * @param filepath (out)
 * @param return
 *		-1:error
 *		0:success
 *
 */
int ConfReader::retrieveFilePath(string &filepath) {
	/*
	string temppath = L"C:\\";
	if (temppath.size() < 1) {
		return -1;
	}
	string temp = FILE_SEPERATOR;
	if (temppath.compare(temppath.size() - 1, 1, FILE_SEPERATOR)) {
		temppath.append(FILE_SEPERATOR);
	}
	temppath.append(INI_FILE_NAME);
	 */

	filepath = this->filePath;
	return 0;
}
