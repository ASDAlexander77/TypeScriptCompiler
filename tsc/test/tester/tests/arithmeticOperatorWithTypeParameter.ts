function foo<T>(t: T) {
    let a: TypeOf<1>;
    let b: boolean;
    let c: number;
    // TODO: if you use uninit. string it will crash the code
    let d: string = "0";

    let r1a1 = a * t;
    let r1a2 = a / t;
    let r1a3 = a % t;
    let r1a4 = a - t;
    let r1a5 = a << t;
    let r1a6 = a >> t;
    let r1a7 = a >>> t;
    let r1a8 = a & t;
    let r1a9 = a ^ t;
    let r1a10 = a | t;

    let r2a1 = t * a;
    let r2a2 = t / a;
    let r2a3 = t % a;
    let r2a4 = t - a;
    let r2a5 = t << a;
    let r2a6 = t >> a;
    let r2a7 = t >>> a;
    let r2a8 = t & a;
    let r2a9 = t ^ a;
    let r2a10 = t | a;

    let r1b1 = b * t;
    let r1b2 = b / t;
    let r1b3 = b % t;
    let r1b4 = b - t;
    let r1b5 = b << t;
    let r1b6 = b >> t;
    let r1b7 = b >>> t;
    let r1b8 = b & t;
    let r1b9 = b ^ t;
    let r1b10 = b | t;

    let r2b1 = t * b;
    let r2b2 = t / b;
    let r2b3 = t % b;
    let r2b4 = t - b;
    let r2b5 = t << b;
    let r2b6 = t >> b;
    let r2b7 = t >>> b;
    let r2b8 = t & b;
    let r2b9 = t ^ b;
    let r2b10 = t | b;

    let r1c1 = c * t;
    let r1c2 = c / t;
    let r1c3 = c % t;
    let r1c4 = c - t;
    let r1c5 = c << t;
    let r1c6 = c >> t;
    let r1c7 = c >>> t;
    let r1c8 = c & t;
    let r1c9 = c ^ t;
    let r1c10 = c | t;

    let r2c1 = t * c;
    let r2c2 = t / c;
    let r2c3 = t % c;
    let r2c4 = t - c;
    let r2c5 = t << c;
    let r2c6 = t >> c;
    let r2c7 = t >>> c;
    let r2c8 = t & c;
    let r2c9 = t ^ c;
    let r2c10 = t | c;

    let r1d1 = d * t;
    let r1d2 = d / t;
    let r1d3 = d % t;
    let r1d4 = d - t;
    let r1d5 = d << t;
    let r1d6 = d >> t;
    let r1d7 = d >>> t;
    let r1d8 = d & t;
    let r1d9 = d ^ t;
    let r1d10 = d | t;

    let r2d1 = t * d;
    let r2d2 = t / d;
    let r2d3 = t % d;
    let r2d4 = t - d;
    let r2d5 = t << d;
    let r2d6 = t >> d;
    let r2d7 = t >>> d;
    let r2d8 = t & d;
    let r2d9 = t ^ d;
    let r2d10 = t | d;

    let r1f1 = t * t;
    let r1f2 = t / t;
    let r1f3 = t % t;
    let r1f4 = t - t;
    let r1f5 = t << t;
    let r1f6 = t >> t;
    let r1f7 = t >>> t;
    let r1f8 = t & t;
    let r1f9 = t ^ t;
    let r1f10 = t | t;
}

function main() {

    foo(1);
    foo(1.0);
    foo(<number>1.0);  

    print("done.");
}