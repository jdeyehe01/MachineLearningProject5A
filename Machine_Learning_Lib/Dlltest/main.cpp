#if _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#include <iostream>

extern "C"
{
	DLLEXPORT int my_add(int x, int y);
}

int main() {
	std::cout << my_add(1, 2) << std::endl;
}