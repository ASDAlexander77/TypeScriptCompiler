function main() {
    let x = 1

    x = 2;
    assert(x == 2);
    x += 2;
    assert(x == 4);
    x -= 3;
    assert(x == 1);
    x *= 4;
    assert(x == 4);
    x = 5;
    x /= 5;
    assert(x == 1);
    x %= 6;
    assert(x == 1);
    x <<= 8;
    assert(x == 256);
    x >>= 7;
    assert(x == 2);
    x >>>= 9;
    assert(x == 0);
    x &= 10;
    assert(x == 0);
    x |= 11;
    assert(x == 11);
    x ^= 12;
    assert(x == 7);

    x++;
    assert(x == 8);
    x--;
    assert(x == 7);
    ++x;
    assert(x == 8);
    --x;
    assert(x == 7);

    ++((x));
    assert(x == 8);

    let a = false;
    a ||= true;
    assert(a);

    let b = true;
    b &&= false;
    assert(!b);

    print("done.");
}
