function main() 
{
    const foo: (number | string)[] = ['aaa'];

    //const isString = (x: unknown): x is string => typeof x === 'string';
    const isString = <T>(x: T): x is string => typeof x === 'string';

    if (foo.every(isString)) {
        print("all strings");
    }
    else
    {
        assert(false);
    }

    print("done.");
}                           