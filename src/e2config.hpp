#ifndef e2config_hpp
#define e2config_hpp

#include <iostream>
#include <map>
#include <string>

namespace e2config {
    struct Map: std::map <std::string, std::string> {
	// Here is a little convenience method...
	bool iskey( const std::string& s ) const {
	    return count(s) != 0;
        }
    };
    std::istream& operator >> ( std::istream& ins, Map& d);
    std::ostream& operator << (std::ostream& outs, const Map& d);

} // namespace e2config


#endif /* e2config_hpp */
