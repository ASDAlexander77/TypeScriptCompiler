// A value of a constructor type of an interface is not supported: a class given to one is an error at the call
// (`new () => A` with `A` a class is the class value `typeof A`, and takes `A` or a subclass).
interface I { }
class A implements I { }
declare function make(c: new () => I): I;
make(A);
