// CRT entry for all platforms: TypeScript exports Main; C++ owns the GLFW loop.

extern "C" void Main();
extern "C" int run_loop();

int main()
{
    Main();
    return run_loop();
}
