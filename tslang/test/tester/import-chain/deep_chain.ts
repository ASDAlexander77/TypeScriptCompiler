// A chain of 20 imports, each module importing the one below it and chain_01 as well. Each
// import was discovered again in every module it was generated into, so a module nested d
// deep was generated 2^d times: this one took minutes and gigabytes, and the TIMEOUT fails it.
import './chain_20'

assert(chain20() == 210);

print("done.");
