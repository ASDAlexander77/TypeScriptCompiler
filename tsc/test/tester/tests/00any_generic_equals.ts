namespace EqualityHelper
{
    function equals<K>(l: K, r: K): boolean {
        if (typeof(l) == "string")
        { 
            return <string>l == <string>r;
        }        

        return l == r;
    }
}

const s = "asd";
const a: any = "bsd";
const n = 10.0;
const an: any = n;

print(EqualityHelper.equals(s, s));
print(EqualityHelper.equals(a, a));
print(EqualityHelper.equals(n, n));

assert(EqualityHelper.equals(a, a));

assert(!EqualityHelper.equals(a, an));

print("done.");