function main() {

    // operands of an enum type are treated as having the primitive type Number.

    enum E {
        a,
        b
    }

    //let a: any;
    let a: TypeOf<1> = 1;
    let b: number = 2;
    let c: E = E.a;

    // operator *
    let ra1 = c * a;
    let ra2 = c * b;
    let ra3 = c * c;
    let ra4 = a * c;
    let ra5 = b * c;
    let ra6 = E.a * a;
    let ra7 = E.a * b;
    let ra8 = E.a * E.b;
    let ra9 = E.a * 1;
    let ra10 = a * E.b;
    let ra11 = b * E.b;
    let ra12 = 1 * E.b;

    // operator /
    let rb1 = c / a;
    let rb2 = c / b;
    let rb3 = c / c;
    let rb4 = a / c;
    let rb5 = b / c;
    let rb6 = E.a / a;
    let rb7 = E.a / b;
    let rb8 = E.a / E.b;
    let rb9 = E.a / 1;
    let rb10 = a / E.b;
    let rb11 = b / E.b;
    let rb12 = 1 / E.b;

    // operator %
    let rc1 = c % a;
    let rc2 = c % b;
    let rc3 = c % c;
    let rc4 = a % c;
    let rc5 = b % c;
    let rc6 = E.a % a;
    let rc7 = E.a % b;
    let rc8 = E.a % E.b;
    let rc9 = E.a % 1;
    let rc10 = a % E.b;
    let rc11 = b % E.b;
    let rc12 = 1 % E.b;

    // operator -
    let rd1 = c - a;
    let rd2 = c - b;
    let rd3 = c - c;
    let rd4 = a - c;
    let rd5 = b - c;
    let rd6 = E.a - a;
    let rd7 = E.a - b;
    let rd8 = E.a - E.b;
    let rd9 = E.a - 1;
    let rd10 = a - E.b;
    let rd11 = b - E.b;
    let rd12 = 1 - E.b;

    // operator <<
    let re1 = c << a;
    let re2 = c << b;
    let re3 = c << c;
    let re4 = a << c;
    let re5 = b << c;
    let re6 = E.a << a;
    let re7 = E.a << b;
    let re8 = E.a << E.b;
    let re9 = E.a << 1;
    let re10 = a << E.b;
    let re11 = b << E.b;
    let re12 = 1 << E.b;

    // operator >>
    let rf1 = c >> a;
    let rf2 = c >> b;
    let rf3 = c >> c;
    let rf4 = a >> c;
    let rf5 = b >> c;
    let rf6 = E.a >> a;
    let rf7 = E.a >> b;
    let rf8 = E.a >> E.b;
    let rf9 = E.a >> 1;
    let rf10 = a >> E.b;
    let rf11 = b >> E.b;
    let rf12 = 1 >> E.b;

    // operator >>>
    let rg1 = c >>> a;
    let rg2 = c >>> b;
    let rg3 = c >>> c;
    let rg4 = a >>> c;
    let rg5 = b >>> c;
    let rg6 = E.a >>> a;
    let rg7 = E.a >>> b;
    let rg8 = E.a >>> E.b;
    let rg9 = E.a >>> 1;
    let rg10 = a >>> E.b;
    let rg11 = b >>> E.b;
    let rg12 = 1 >>> E.b;

    // operator &
    let rh1 = c & a;
    let rh2 = c & b;
    let rh3 = c & c;
    let rh4 = a & c;
    let rh5 = b & c;
    let rh6 = E.a & a;
    let rh7 = E.a & b;
    let rh8 = E.a & E.b;
    let rh9 = E.a & 1;
    let rh10 = a & E.b;
    let rh11 = b & E.b;
    let rh12 = 1 & E.b;

    // operator ^
    let ri1 = c ^ a;
    let ri2 = c ^ b;
    let ri3 = c ^ c;
    let ri4 = a ^ c;
    let ri5 = b ^ c;
    let ri6 = E.a ^ a;
    let ri7 = E.a ^ b;
    let ri8 = E.a ^ E.b;
    let ri9 = E.a ^ 1;
    let ri10 = a ^ E.b;
    let ri11 = b ^ E.b;
    let ri12 = 1 ^ E.b;

    // operator |
    let rj1 = c | a;
    let rj2 = c | b;
    let rj3 = c | c;
    let rj4 = a | c;
    let rj5 = b | c;
    let rj6 = E.a | a;
    let rj7 = E.a | b;
    let rj8 = E.a | E.b;
    let rj9 = E.a | 1;
    let rj10 = a | E.b;
    let rj11 = b | E.b;
    let rj12 = 1 | E.b;

    print("done.");
}