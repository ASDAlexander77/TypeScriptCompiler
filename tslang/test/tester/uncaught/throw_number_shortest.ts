// nothing catches it: the error says the number as the program prints it, `1e-7`, not `%.17g`'s
// 9.9999999999999995e-08
function main() {
    print("before");
    const tenth: number = 0.1;
    throw tenth * 1e-6;
}
