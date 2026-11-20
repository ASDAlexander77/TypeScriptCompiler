// A value of a constructor type is not supported: a class given to one is an error at the call.
class A { }
declare function make(c: new () => A): A;
make(A);
