// The host of a tslang wasm program (wasm32-unknown-unknown or wasm64-unknown-unknown).
//
// These targets have no C library. tslang links a wasm program with --allow-undefined, so what the C
// library and the default library's C++ wrappers provide on other targets (malloc, snprintf, puts,
// regexp_*, stdio_write, the time functions, ...) are imports from the `env` module, implemented
// here in JavaScript. One module for Node (run.mjs) and the browser (index.html): Node-only parts
// (stdin, fs, process) are used only when `node` is passed in.
//
// Pointers and size_t are i32 on wasm32 and i64 on wasm64, which JavaScript sees as a Number and a
// BigInt. Every import takes them through N() and returns them through P(); C's `long long` (time_t,
// tslang's `long`) is an i64 on both and so always a BigInt.
//
//   const host = createHost({ args, write, node });
//   const { instance } = await WebAssembly.instantiate(bytes, host.imports);
//   const exitCode = host.run(instance);

export class ExitSignal extends Error
{
    constructor(code)
    {
        super(`exit(${code})`);
        this.code = code;
    }
}

// a C++ exception nothing caught: on wasm try/catch does not catch yet, so every throw ends here
export class UncaughtThrow extends Error
{
    constructor(objectAddress)
    {
        super("uncaught exception");
        this.objectAddress = objectAddress;
    }
}

const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

export function createHost(options = {})
{
    const args = options.args ?? ["program.wasm"];
    const write = options.write ?? ((fd, text) => (fd === 2 ? console.error : console.log)(text));
    const node = options.node; // { fs, process } under Node

    let instance;
    let memory;
    let is64 = false;

    const N = (v) => Number(v);
    const P = (v) => (is64 ? BigInt(v) : v);

    const u8 = () => new Uint8Array(memory.buffer);
    const dv = () => new DataView(memory.buffer);

    const utf8Decoder = new TextDecoder("utf-8");
    const utf8Encoder = new TextEncoder();

    const cstrEnd = (p) => { const m = u8(); while (m[p] !== 0) p++; return p; };
    const bytesOf = (p, n) => u8().slice(p, p + n);
    const readStr = (p) => { p = N(p); return p === 0 ? "" : utf8Decoder.decode(u8().subarray(p, cstrEnd(p))); };
    // one char per byte: the regex wrappers count positions in bytes, as std::cmatch does over a char*
    const readLatin1 = (p) => { p = N(p); let s = ""; const m = u8(); for (let e = cstrEnd(p); p < e; p++) s += String.fromCharCode(m[p]); return s; };
    const latin1Bytes = (s) => { const b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i) & 0xff; return b; };
    // `bytes` and a 0 at `p`, at most `max` bytes with the 0
    const writeBytesZ = (p, bytes, max = Infinity) =>
    {
        p = N(p);
        if (max <= 0) return 0;
        const n = Math.min(bytes.length, max - 1);
        u8().set(bytes.subarray(0, n), p);
        u8()[p + n] = 0;
        return n;
    };
    const writeStrZ = (p, s, max) => writeBytesZ(p, utf8Encoder.encode(s), max);

    // ---- the heap: free lists per 16-byte size class above a bump pointer, the memory grown as needed ----

    const ALIGN = 16;
    let heapTop = 0;
    const sizes = new Map();
    const freeLists = new Map();

    const ensure = (end) =>
    {
        const have = memory.buffer.byteLength;
        if (end <= have) return;
        const pages = Math.ceil((end - have) / 65536);
        memory.grow(is64 ? BigInt(pages) : pages);
    };

    const malloc = (size) =>
    {
        size = Math.max(ALIGN, Math.ceil(N(size) / ALIGN) * ALIGN);
        const list = freeLists.get(size);
        let p;
        if (list && list.length)
        {
            p = list.pop();
        }
        else
        {
            p = heapTop;
            heapTop += size;
            ensure(heapTop);
        }

        sizes.set(p, size);
        return p;
    };

    const free = (p) =>
    {
        p = N(p);
        if (p === 0) return;
        const size = sizes.get(p);
        if (size === undefined) throw new Error(`free of ${p}: not a block the heap gave out`);
        sizes.delete(p);
        let list = freeLists.get(size);
        if (!list) freeLists.set(size, list = []);
        list.push(p);
    };

    const realloc = (p, size) =>
    {
        p = N(p);
        size = N(size);
        if (p === 0) return malloc(size);
        const old = sizes.get(p);
        if (old === undefined) throw new Error(`realloc of ${p}: not a block the heap gave out`);
        if (size <= old) return p;
        const q = malloc(size);
        u8().copyWithin(q, p, p + old);
        free(p);
        return q;
    };

    const calloc = (count, size) =>
    {
        const n = N(count) * N(size);
        const p = malloc(n);
        u8().fill(0, p, p + n);
        return p;
    };

    // ---- printf ----

    // C's %e: at least two exponent digits
    const expForm = (v, prec, upper) =>
    {
        let s = v.toExponential(prec).replace(/e([+-])(\d)$/, "e$10$2");
        return upper ? s.toUpperCase() : s;
    };

    const stripZeros = (s) =>
    {
        if (s.indexOf(".") < 0) return s;
        const e = s.search(/[eE]/);
        let mantissa = e < 0 ? s : s.slice(0, e);
        const exp = e < 0 ? "" : s.slice(e);
        mantissa = mantissa.replace(/0+$/, "").replace(/\.$/, "");
        return mantissa + exp;
    };

    const fixedForm = (v, prec) =>
    {
        if (prec <= 100 && Math.abs(v) < 1e21) return v.toFixed(prec);
        // toFixed gives exponent notation from 1e21 on; C prints every digit
        let s = BigInt(Math.trunc(v)).toString();
        if (v < 0 && s[0] !== "-") s = "-" + s;
        return prec > 0 ? s + "." + "0".repeat(prec) : s;
    };

    const hexFloat = (v, prec, upper) =>
    {
        if (v === 0) return (Object.is(v, -0) ? "-" : "") + "0x0p+0";
        const sign = v < 0 ? "-" : "";
        v = Math.abs(v);
        const buf = new DataView(new ArrayBuffer(8));
        buf.setFloat64(0, v);
        const bits = buf.getBigUint64(0);
        let exp = Number((bits >> 52n) & 0x7ffn);
        let mant = bits & 0xfffffffffffffn;
        let lead = 1n;
        if (exp === 0) { lead = 0n; exp = -1022; } else exp -= 1023;
        let hex = mant.toString(16).padStart(13, "0");
        if (prec === undefined) hex = hex.replace(/0+$/, "");
        else hex = hex.slice(0, prec).padEnd(prec, "0");
        const s = `${sign}0x${lead}${hex ? "." + hex : ""}p${exp >= 0 ? "+" : ""}${exp}`;
        return upper ? s.toUpperCase().replace("0X", "0X") : s;
    };

    const formatFloat = (v, conv, prec, alt) =>
    {
        const upper = conv === conv.toUpperCase();
        if (!isFinite(v))
        {
            const s = isNaN(v) ? "nan" : (v < 0 ? "-inf" : "inf");
            return upper ? s.toUpperCase() : s;
        }

        switch (conv.toLowerCase())
        {
            case "f": return fixedForm(v, prec ?? 6);
            case "e": return expForm(v, prec ?? 6, upper);
            case "a": return hexFloat(v, prec, upper);
            case "g":
            {
                let p = prec ?? 6;
                if (p === 0) p = 1;
                if (v === 0) return alt ? (0).toFixed(p - 1) : (Object.is(v, -0) ? "-0" : "0");
                const x = Number(v.toExponential(p - 1).split("e")[1]);
                let s = (p > x && x >= -4) ? fixedForm(v, p - 1 - x) : expForm(v, p - 1, upper);
                return alt ? s : stripZeros(s);
            }
        }

        return String(v);
    };

    // C's printf over a format and the wasm vararg buffer: each argument at its own alignment
    const format = (fmt, va) =>
    {
        va = N(va);
        const view = dv();
        const ptrSize = is64 ? 8 : 4;
        const take = (size, kind) =>
        {
            va = Math.ceil(va / size) * size;
            let v;
            if (kind === "f") v = view.getFloat64(va, true);
            else if (size === 8) v = kind === "u" ? view.getBigUint64(va, true) : view.getBigInt64(va, true);
            else v = kind === "u" ? view.getUint32(va, true) : view.getInt32(va, true);
            va += size;
            return v;
        };

        let out = "";
        const re = /%([-+ #0]*)(\*|\d+)?(?:\.(\*|\d*))?(hh|h|ll|l|L|z|j|t|q)?([diuoxXcspfFeEgGaAn%])/g;
        let last = 0;
        let m;
        while ((m = re.exec(fmt)) !== null)
        {
            out += fmt.slice(last, m.index);
            last = re.lastIndex;
            let [, flags, width, prec, len, conv] = m;
            if (conv === "%") { out += "%"; continue; }
            if (width === "*") width = take(4, "i");
            if (prec === "*") prec = take(4, "i");
            width = width === undefined ? 0 : Number(width);
            if (width < 0) { flags += "-"; width = -width; }
            prec = prec === undefined ? undefined : (prec === "" ? 0 : Number(prec));
            if (prec !== undefined && prec < 0) prec = undefined;

            const intSize = (len === "ll" || len === "q" || len === "j") ? 8 : (len === "l" || len === "z" || len === "t") ? ptrSize : 4;
            let body;
            let sign = "";
            switch (conv)
            {
                case "d":
                case "i":
                {
                    let v = BigInt(take(intSize, "i"));
                    if (len === "h") v = BigInt.asIntN(16, v);
                    if (len === "hh") v = BigInt.asIntN(8, v);
                    if (v < 0n) { sign = "-"; v = -v; }
                    else if (flags.includes("+")) sign = "+";
                    else if (flags.includes(" ")) sign = " ";
                    body = v.toString();
                    if (prec !== undefined) body = (prec === 0 && v === 0n) ? "" : body.padStart(prec, "0");
                    break;
                }
                case "u":
                case "o":
                case "x":
                case "X":
                {
                    let v = BigInt(take(intSize, "u"));
                    v = BigInt.asUintN(intSize * 8, v);
                    if (len === "h") v = BigInt.asUintN(16, v);
                    if (len === "hh") v = BigInt.asUintN(8, v);
                    body = v.toString(conv === "u" ? 10 : conv === "o" ? 8 : 16);
                    if (conv === "X") body = body.toUpperCase();
                    if (prec !== undefined) body = (prec === 0 && v === 0n) ? "" : body.padStart(prec, "0");
                    if (flags.includes("#") && v !== 0n)
                    {
                        if (conv === "x") sign = "0x";
                        else if (conv === "X") sign = "0X";
                        else if (conv === "o" && body[0] !== "0") body = "0" + body;
                    }
                    break;
                }
                case "c":
                    body = String.fromCharCode(take(4, "i") & 0xff);
                    break;
                case "s":
                {
                    const p = Number(take(ptrSize, "u"));
                    body = p === 0 ? "(null)" : readStr(p);
                    if (prec !== undefined) body = body.slice(0, prec);
                    break;
                }
                case "p":
                    body = "0x" + BigInt(take(ptrSize, "u")).toString(16);
                    break;
                case "n":
                    take(ptrSize, "u");
                    body = "";
                    break;
                default:
                {
                    let v = take(8, "f");
                    body = formatFloat(v, conv, prec, flags.includes("#"));
                    if (body[0] === "-") { sign = "-"; body = body.slice(1); }
                    else if (flags.includes("+")) sign = "+";
                    else if (flags.includes(" ")) sign = " ";
                    break;
                }
            }

            const length = sign.length + body.length;
            if (length < width)
            {
                if (flags.includes("-")) body = body + " ".repeat(width - length);
                else if (flags.includes("0") && conv !== "s" && conv !== "c" && !(prec !== undefined && "diuoxX".includes(conv)) && /^[0-9a-fA-F.]/.test(body))
                    body = "0".repeat(width - length) + body;
                else sign = " ".repeat(width - length) + sign;
            }

            out += sign + body;
        }

        return out + fmt.slice(last);
    };

    // ---- strtod / strtol ----

    const strtod = (str, endPtr) =>
    {
        const p = N(str);
        const s = readLatin1(p);
        const m = /^[ \t\n\v\f\r]*([+-]?(?:0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?\d+)?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan(?:\([0-9A-Za-z_]*\))?))/i.exec(s);
        let value = 0;
        let used = 0;
        if (m)
        {
            used = m[0].length;
            let t = m[1];
            let sign = 1;
            if (t[0] === "-" || t[0] === "+") { if (t[0] === "-") sign = -1; t = t.slice(1); }
            if (/^inf/i.test(t)) value = Infinity;
            else if (/^nan/i.test(t)) value = NaN;
            else if (/^0x/i.test(t))
            {
                const hm = /^0x([0-9a-f]*)(?:\.([0-9a-f]*))?(?:p([+-]?\d+))?$/i.exec(t);
                const intPart = hm[1] || "0";
                const frac = hm[2] || "";
                value = parseInt(intPart, 16) + (frac ? parseInt(frac, 16) / Math.pow(16, frac.length) : 0);
                value *= Math.pow(2, Number(hm[3] || 0));
            }
            else value = Number(t);
            value *= sign;
        }

        if (N(endPtr) !== 0)
        {
            const view = dv();
            const e = p + used;
            if (is64) view.setBigUint64(N(endPtr), BigInt(e), true); else view.setUint32(N(endPtr), e, true);
        }

        return value;
    };

    const strtol = (str, endPtr, base) =>
    {
        const p = N(str);
        const s = readLatin1(p);
        let i = 0;
        while (/[ \t\n\v\f\r]/.test(s[i] ?? "")) i++;
        let sign = 1n;
        if (s[i] === "+" || s[i] === "-") { if (s[i] === "-") sign = -1n; i++; }
        if ((base === 0 || base === 16) && s[i] === "0" && /[xX]/.test(s[i + 1] ?? "") && /[0-9a-fA-F]/.test(s[i + 2] ?? "")) { i += 2; base = 16; }
        else if (base === 0 && s[i] === "0") base = 8;
        else if (base === 0) base = 10;
        const start = i;
        let v = 0n;
        for (; i < s.length; i++)
        {
            const d = parseInt(s[i], 36);
            if (isNaN(d) || d >= base) break;
            v = v * BigInt(base) + BigInt(d);
        }

        const used = i === start ? 0 : i;
        if (N(endPtr) !== 0)
        {
            const view = dv();
            if (is64) view.setBigUint64(N(endPtr), BigInt(p + used), true); else view.setUint32(N(endPtr), p + used, true);
        }

        v *= sign;
        if (v > 2147483647n) return 2147483647;
        if (v < -2147483648n) return -2147483648;
        return Number(v);
    };

    // strtoll / strtoull: strtol's parse, clamped to 64 bits; a BigInt
    const strtolWide = (str, endPtr, base, signed) =>
    {
        const p = N(str);
        const s = readLatin1(p);
        const m = /^[ \t\n\v\f\r]*([+-]?)(0[xX](?=[0-9a-fA-F]))?([0-9a-zA-Z]*)/.exec(s);
        let radix = base;
        let digits = m[3];
        if (m[2] && (radix === 0 || radix === 16)) radix = 16;
        else if (radix === 0) radix = digits[0] === "0" ? 8 : 10;
        let v = 0n, used = 0;
        let i = 0;
        for (; i < digits.length; i++)
        {
            const d = parseInt(digits[i], 36);
            if (isNaN(d) || d >= radix) break;
            v = v * BigInt(radix) + BigInt(d);
        }

        if (i > 0) used = m[0].length - (digits.length - i);
        if (m[1] === "-") v = -v;
        if (N(endPtr) !== 0)
        {
            const view = dv();
            if (is64) view.setBigUint64(N(endPtr), BigInt(p + used), true); else view.setUint32(N(endPtr), p + used, true);
        }

        if (signed) return v > 0x7fffffffffffffffn ? 0x7fffffffffffffffn : v < -0x8000000000000000n ? -0x8000000000000000n : v;
        return BigInt.asUintN(64, v);
    };

    // a string that stays: strerror's result
    const staticStrings = new Map();
    const staticString = (s) =>
    {
        if (!staticStrings.has(s)) staticStrings.set(s, mallocBytesZ(utf8Encoder.encode(s)));
        return staticStrings.get(s);
    };

    // ---- time ----

    const readI64 = (p) => dv().getBigInt64(N(p), true);
    const TM_FIELDS = 9;
    const readTm = (p) => { p = N(p); const v = dv(); const t = []; for (let i = 0; i < TM_FIELDS; i++) t.push(v.getInt32(p + i * 4, true)); return t; };
    const writeTm = (p, t) => { p = N(p); const v = dv(); for (let i = 0; i < TM_FIELDS; i++) v.setInt32(p + i * 4, t[i], true); };
    // [sec, min, hour, mday, mon, year - 1900, wday, yday, isdst]
    const tmOf = (date, utc) =>
    {
        const g = (name) => date[(utc ? "getUTC" : "get") + name]();
        const year = g("FullYear");
        const startOfYear = utc ? Date.UTC(year, 0, 1) : new Date(year, 0, 1).getTime();
        const startOfDay = utc ? Date.UTC(year, g("Month"), g("Date")) : new Date(year, g("Month"), g("Date")).getTime();
        const yday = Math.round((startOfDay - startOfYear) / 86400000);
        return [g("Seconds"), g("Minutes"), g("Hours"), g("Date"), g("Month"), year - 1900, g("Day"), yday, utc ? 0 : (isDst(date) ? 1 : 0)];
    };
    const standardOffset = (year) => Math.max(new Date(year, 0, 1).getTimezoneOffset(), new Date(year, 6, 1).getTimezoneOffset());
    const isDst = (date) => date.getTimezoneOffset() < standardOffset(date.getFullYear());
    const dateOfTm = (t, utc) =>
    {
        const [sec, min, hour, mday, mon, year] = t;
        const d = new Date(0);
        if (utc) { d.setUTCFullYear(year + 1900, mon, mday); d.setUTCHours(hour, min, sec, 0); }
        else { d.setFullYear(year + 1900, mon, mday); d.setHours(hour, min, sec, 0); }
        return d;
    };
    const pad = (n, w = 2, c = "0") => String(n).padStart(w, c);
    const asctime = (t) => `${WEEKDAYS[t[6]].slice(0, 3)} ${MONTHS[t[4]].slice(0, 3)} ${pad(t[3], 2, " ")} ${pad(t[2])}:${pad(t[1])}:${pad(t[0])} ${t[5] + 1900}\n`;
    const zoneOf = (t) =>
    {
        // a tm from gmtime has tm_isdst 0 and is read as UTC by glibc's %z/%Z only with tm_gmtoff;
        // here every tm is local unless it is the UTC one gmtime_r gave
        const d = dateOfTm(t, false);
        const offset = -d.getTimezoneOffset();
        const sign = offset < 0 ? "-" : "+";
        const z = `${sign}${pad(Math.floor(Math.abs(offset) / 60))}${pad(Math.abs(offset) % 60)}`;
        let name = "";
        try { name = new Intl.DateTimeFormat("en-US", { timeZoneName: "short" }).formatToParts(d).find((x) => x.type === "timeZoneName")?.value ?? ""; } catch { }
        return { z, name };
    };
    const strftime = (fmt, t) =>
    {
        const [sec, min, hour, mday, mon, year, wday, yday] = t;
        const Y = year + 1900;
        return fmt.replace(/%([EO]?)([a-zA-Z%])/g, (all, mod, c) =>
        {
            switch (c)
            {
                case "a": return WEEKDAYS[wday].slice(0, 3);
                case "A": return WEEKDAYS[wday];
                case "b": case "h": return MONTHS[mon].slice(0, 3);
                case "B": return MONTHS[mon];
                case "c": return asctime(t).slice(0, -1);
                case "C": return pad(Math.floor(Y / 100));
                case "d": return pad(mday);
                case "D": return `${pad(mon + 1)}/${pad(mday)}/${pad(Y % 100)}`;
                case "e": return pad(mday, 2, " ");
                case "F": return `${Y}-${pad(mon + 1)}-${pad(mday)}`;
                case "H": return pad(hour);
                case "I": return pad(hour % 12 === 0 ? 12 : hour % 12);
                case "j": return pad(yday + 1, 3);
                case "k": return pad(hour, 2, " ");
                case "l": return pad(hour % 12 === 0 ? 12 : hour % 12, 2, " ");
                case "m": return pad(mon + 1);
                case "M": return pad(min);
                case "n": return "\n";
                case "p": return hour < 12 ? "AM" : "PM";
                case "P": return hour < 12 ? "am" : "pm";
                case "r": return `${pad(hour % 12 === 0 ? 12 : hour % 12)}:${pad(min)}:${pad(sec)} ${hour < 12 ? "AM" : "PM"}`;
                case "R": return `${pad(hour)}:${pad(min)}`;
                case "S": return pad(sec);
                case "t": return "\t";
                case "T": return `${pad(hour)}:${pad(min)}:${pad(sec)}`;
                case "u": return String(wday === 0 ? 7 : wday);
                case "w": return String(wday);
                case "x": return `${pad(mon + 1)}/${pad(mday)}/${pad(Y % 100)}`;
                case "X": return `${pad(hour)}:${pad(min)}:${pad(sec)}`;
                case "y": return pad(Y % 100);
                case "Y": return String(Y);
                case "z": return zoneOf(t).z;
                case "Z": return zoneOf(t).name;
                case "%": return "%";
                default: return all;
            }
        });
    };

    // ---- regex: std::regex's results, by handle ----

    const matches = new Map();
    let nextMatch = 1;
    const jsFlags = (flags) => "d" + (flags.includes("i") ? "i" : "") + (flags.includes("m") ? "m" : "");
    const regexError = (e) => { write(2, `Some other regex exception happened.\n${e.message}\n`); };
    // std::match_results::format, ECMAScript rules: $&, $n, $nn, $`, $', $$
    const formatMatch = (m, s, replacement) =>
        replacement.replace(/\$(\$|&|`|'|\d{1,2})/g, (all, x) =>
        {
            if (x === "$") return "$";
            if (x === "&") return m[0];
            if (x === "`") return s.slice(0, m.index);
            if (x === "'") return s.slice(m.index + m[0].length);
            let n = Number(x);
            if (x.length === 2 && n >= m.length) { n = Number(x[0]); return (n < m.length ? (m[n] ?? "") : all.slice(0, 2)) + x[1]; }
            return n < m.length ? (m[n] ?? "") : all;
        });
    const mallocBytesZ = (bytes) => { const p = malloc(bytes.length + 1); writeBytesZ(p, bytes, bytes.length + 1); return p; };

    // ---- http: no network from a wasm program; every request fails ----

    const responses = new Map();
    let nextResponse = 1;

    // ---- process and fs (Node) ----

    let result = "";
    let resultBytes = new Uint8Array(0);
    let entries = [];
    const setResult = (bytes) => { resultBytes = typeof bytes === "string" ? utf8Encoder.encode(bytes) : bytes; return resultBytes.length; };
    const fsFail = (e, op, path, dest) =>
    {
        const code = e?.code ?? "EIO";
        const text = (e?.message ?? String(e)).replace(/^[A-Z]+: /, "").replace(/, .*$/, "");
        setResult(`${code}: ${text}, ${op}${path !== undefined ? ` '${path}'` : ""}${dest !== undefined ? ` -> '${dest}'` : ""}`);
        return -1;
    };
    const needNode = (what) => { if (!node) throw new Error(`${what} needs the Node host`); };

    let stdinLine = new Uint8Array(0);
    const stdinBuffer = { bytes: new Uint8Array(0), pos: 0, eof: false };
    const stdinByte = () =>
    {
        if (stdinBuffer.pos >= stdinBuffer.bytes.length)
        {
            if (stdinBuffer.eof || !node) return -1;
            const chunk = new Uint8Array(65536);
            let n = 0;
            try { n = node.fs.readSync(0, chunk, 0, chunk.length, null); } catch (e) { if (e.code === "EAGAIN") return stdinByte(); if (e.code !== "EOF") n = 0; }
            if (n <= 0) { stdinBuffer.eof = true; return -1; }
            stdinBuffer.bytes = chunk.subarray(0, n);
            stdinBuffer.pos = 0;
        }

        return stdinBuffer.bytes[stdinBuffer.pos++];
    };

    const env = {
        // memory
        malloc: (size) => P(malloc(size)),
        calloc: (count, size) => P(calloc(count, size)),
        realloc: (p, size) => P(realloc(p, size)),
        free: (p) => free(p),
        boundedCopy: (dest, destSize, src, count) => { const n = Math.min(N(destSize), N(count)); u8().copyWithin(N(dest), N(src), N(src) + n); },
        memset: (p, value, n) => { u8().fill(value & 0xff, N(p), N(p) + N(n)); return p; },
        memcpy: (d, s, n) => { u8().copyWithin(N(d), N(s), N(s) + N(n)); return d; },
        memmove: (d, s, n) => { u8().copyWithin(N(d), N(s), N(s) + N(n)); return d; },
        memcmp: (a, b, n) => { const m = u8(); a = N(a); b = N(b); for (let i = 0; i < N(n); i++) if (m[a + i] !== m[b + i]) return m[a + i] - m[b + i]; return 0; },

        // strings
        strlen: (p) => P(cstrEnd(N(p)) - N(p)),
        strcmp: (a, b) =>
        {
            const m = u8();
            a = N(a); b = N(b);
            while (m[a] !== 0 && m[a] === m[b]) { a++; b++; }
            return m[a] - m[b];
        },
        strncmp: (a, b, n) =>
        {
            const m = u8();
            a = N(a); b = N(b); n = N(n);
            for (let i = 0; i < n; i++, a++, b++) { if (m[a] !== m[b] || m[a] === 0) return m[a] - m[b]; }
            return 0;
        },
        strcoll: (a, b) => { const x = readStr(a), y = readStr(b); return x < y ? -1 : x > y ? 1 : 0; },
        strchr: (p, c) =>
        {
            const m = u8();
            p = N(p); c &= 0xff;
            for (; ; p++) { if (m[p] === c) return P(p); if (m[p] === 0) return P(0); }
        },
        strstr: (h, n) =>
        {
            const hs = readLatin1(h), ns = readLatin1(n);
            const i = hs.indexOf(ns);
            return P(i < 0 ? 0 : N(h) + i);
        },
        strcpy: (d, s) => { const n = cstrEnd(N(s)) - N(s); u8().copyWithin(N(d), N(s), N(s) + n + 1); return d; },
        strcat: (d, s) => { const e = cstrEnd(N(d)); const n = cstrEnd(N(s)) - N(s); u8().copyWithin(e, N(s), N(s) + n + 1); return d; },
        tolower: (c) => (c >= 65 && c <= 90 ? c + 32 : c),
        toupper: (c) => (c >= 97 && c <= 122 ? c - 32 : c),
        isspace: (c) => ((c === 32 || (c >= 9 && c <= 13)) ? 1 : 0),
        isdigit: (c) => ((c >= 48 && c <= 57) ? 1 : 0),
        isalpha: (c) => (((c | 32) >= 97 && (c | 32) <= 122) ? 1 : 0),
        atoi: (p) => strtol(p, P(0), 10),
        atof: (p) => strtod(p, P(0)),
        strtod,
        strtol,
        strtoll: (str, endPtr, base) => BigInt(strtolWide(str, endPtr, base, true)),
        strtoull: (str, endPtr, base) => BigInt.asIntN(64, strtolWide(str, endPtr, base, false)),
        strerror: (errnum) => P(staticString(`errno ${errnum}`)),
        getenv: (name) => { const v = node?.process.env[readStr(name)]; return P(v === undefined ? 0 : staticString(v)); },
        qsort: (base, count, size, compare) =>
        {
            base = N(base); count = N(count); size = N(size);
            const cmp = instance.exports.__indirect_function_table.get(N(compare));
            const items = [];
            for (let i = 0; i < count; i++) items.push(u8().slice(base + i * size, base + (i + 1) * size));
            // each pair is compared in two scratch blocks, as C's qsort hands out pointers into the array
            const a = malloc(size), b = malloc(size);
            items.sort((x, y) => { u8().set(x, a); u8().set(y, b); return cmp(P(a), P(b)); });
            free(a); free(b);
            items.forEach((item, i) => u8().set(item, base + i * size));
        },
        snprintf: (out, n, fmt, va) =>
        {
            const s = format(readStr(fmt), va);
            const bytes = utf8Encoder.encode(s);
            if (N(out) !== 0 && N(n) > 0) writeBytesZ(out, bytes, N(n));
            return bytes.length;
        },
        sprintf: (out, fmt, va) => { const bytes = utf8Encoder.encode(format(readStr(fmt), va)); return writeBytesZ(out, bytes); },
        printf: (fmt, va) => { const s = format(readStr(fmt), va); write(1, s); return s.length; },
        setlocale: () => P(0),

        // output
        puts: (p) => { write(1, readStr(p) + "\n"); return 1; },
        putchar: (c) => { write(1, String.fromCharCode(c)); return c; },
        fflush: () => 0,
        stdio_write: (fd, s, count) => { if (count > 0) write(fd === 2 ? 2 : 1, utf8Decoder.decode(bytesOf(N(s), count))); },
        stdio_write_bytes: (fd, s, count) => { if (count > 0) write(fd === 2 ? 2 : 1, utf8Decoder.decode(bytesOf(N(s), count))); return Math.max(0, count); },
        stdio_read_line: () =>
        {
            const bytes = [];
            let c;
            let read = false;
            while ((c = stdinByte()) >= 0) { read = true; if (c === 10) break; bytes.push(c); }
            if (!read) return -1;
            if (bytes.length && bytes[bytes.length - 1] === 13) bytes.pop();
            stdinLine = Uint8Array.from(bytes);
            return stdinLine.length;
        },
        stdio_line_copy_to: (buffer, count) => { if (N(buffer) && count > 0) u8().set(stdinLine.subarray(0, count), N(buffer)); },
        stdio_read: (buffer, count) =>
        {
            let n = 0;
            const m = u8();
            for (; n < count; n++) { const c = stdinByte(); if (c < 0) break; m[N(buffer) + n] = c; }
            return n;
        },

        // C++ exceptions: nothing catches them on wasm yet
        __cxa_allocate_exception: (size) => P(malloc(size)),
        __cxa_free_exception: (p) => free(p),
        __cxa_throw: (object) => { throw new UncaughtThrow(N(object)); },
        __assert_fail: (msg, file, line, fn) => { throw new Error(`Assertion failed: ${readStr(msg)}, file ${readStr(file)}, line ${line}`); },
        abort: () => { throw new Error("abort()"); },
        exit: (code) => { throw new ExitSignal(code); },

        // math
        acos: Math.acos, acosh: Math.acosh, asin: Math.asin, asinh: Math.asinh, atan: Math.atan, atan2: Math.atan2, atanh: Math.atanh,
        cbrt: Math.cbrt, cos: Math.cos, cosh: Math.cosh, exp: Math.exp, expm1: Math.expm1, log: Math.log, log10: Math.log10,
        log1p: Math.log1p, log2: Math.log2, pow: Math.pow, sin: Math.sin, sinh: Math.sinh, tan: Math.tan, tanh: Math.tanh,
        sqrt: Math.sqrt, floor: Math.floor, ceil: Math.ceil, trunc: Math.trunc, fabs: Math.abs, hypot: Math.hypot,
        // C's round: halves away from zero (Math.round takes them up)
        round: (x) => (x < 0 ? -Math.round(-x) : Math.round(x)),
        fmod: (x, y) => x % y,
        fmin: Math.min, fmax: Math.max,
        abs: (x) => (x < 0 ? -x | 0 : x),
        labs: (x) => (typeof x === "bigint" ? (x < 0n ? -x : x) : (x < 0 ? -x | 0 : x)),
        llabs: (x) => (x < 0n ? -x : x),
        rand: () => Math.floor(Math.random() * 2147483648),
        srand: () => { },

        // time
        time: (p) => { const t = BigInt(Math.floor(Date.now() / 1000)); if (N(p) !== 0) dv().setBigInt64(N(p), t, true); return t; },
        tslang_wasm_now: () => Date.now(),
        tslang_wasm_timezone: () => standardOffset(new Date().getFullYear()) * 60,
        gmtime_r: (timePtr, tmPtr) => { writeTm(tmPtr, tmOf(new Date(Number(readI64(timePtr)) * 1000), true)); return tmPtr; },
        localtime_r: (timePtr, tmPtr) => { writeTm(tmPtr, tmOf(new Date(Number(readI64(timePtr)) * 1000), false)); return tmPtr; },
        timegm: (tmPtr) => { const d = dateOfTm(readTm(tmPtr), true); return isNaN(d.getTime()) ? -1n : BigInt(Math.floor(d.getTime() / 1000)); },
        mktime: (tmPtr) => { const d = dateOfTm(readTm(tmPtr), false); return isNaN(d.getTime()) ? -1n : BigInt(Math.floor(d.getTime() / 1000)); },
        ctime_r: (timePtr, buffer) => { writeStrZ(buffer, asctime(tmOf(new Date(Number(readI64(timePtr)) * 1000), false)), 26); return buffer; },
        asctime_r: (tmPtr, buffer) => { writeStrZ(buffer, asctime(readTm(tmPtr)), 26); return buffer; },
        strftime: (out, max, fmt, tmPtr) =>
        {
            const bytes = utf8Encoder.encode(strftime(readStr(fmt), readTm(tmPtr)));
            if (bytes.length + 1 > N(max)) { if (N(max) > 0) u8()[N(out)] = 0; return P(0); }
            return P(writeBytesZ(out, bytes, N(max)));
        },
        parse_date: (p) => { const t = Date.parse(readStr(p)); return isNaN(t) ? -1n : BigInt(t); },

        // regex (regex.cpp)
        regexp_test: (expr, flags, s) =>
        {
            try
            {
                const m = new RegExp(readLatin1(expr), jsFlags(readStr(flags))).exec(readLatin1(s));
                return m ? m.index + m[0].length : -1;
            }
            catch (e) { regexError(e); return -1; }
        },
        regexp_exec: (expr, flags, s, cm) =>
        {
            try
            {
                const text = readLatin1(s);
                const m = new RegExp(readLatin1(expr), jsFlags(readStr(flags))).exec(text);
                if (!m) return P(0);
                let handle = N(cm);
                if (handle === 0) handle = nextMatch++;
                matches.set(handle, { m, text });
                return P(handle);
            }
            catch (e) { regexError(e); return P(0); }
        },
        regexp_match_results_size: (cm) => P(matches.get(N(cm)).m.length),
        regexp_match_results_prefix_length: (cm) => P(matches.get(N(cm)).m.index),
        regexp_match_results_suffix_position: (cm) => { const { m } = matches.get(N(cm)); return P(m.index + m[0].length); },
        regexp_match_results_sub_match_str_length: (cm, i) => P((matches.get(N(cm)).m[N(i)] ?? "").length),
        regexp_match_results_sub_match_str_copy_to: (cm, i, buffer, count) => { const b = latin1Bytes(matches.get(N(cm)).m[N(i)] ?? ""); u8().set(b.subarray(0, N(count)), N(buffer)); },
        regexp_match_results_sub_match_position: (cm, i) => { const { m } = matches.get(N(cm)); const ix = m.indices[N(i)]; return P(ix ? ix[0] : m.index + m[0].length); },
        regexp_match_results_sub_match_length: (cm, i) => P((matches.get(N(cm)).m[N(i)] ?? "").length),
        regexp_match_results_format: (cm, replacement) => { const { m, text } = matches.get(N(cm)); return P(mallocBytesZ(latin1Bytes(formatMatch(m, text, readLatin1(replacement))))); },
        regexp_replace: (expr, flags, s, replacement) =>
        {
            try
            {
                const text = readLatin1(s);
                const repl = readLatin1(replacement);
                const re = new RegExp(readLatin1(expr), "g" + jsFlags(readStr(flags)));
                const result = text.replace(re, (...a) =>
                {
                    const groupsAt = typeof a[a.length - 1] === "object" ? a.length - 3 : a.length - 2;
                    const m = a.slice(0, groupsAt);
                    m.index = a[groupsAt];
                    return formatMatch(m, text, repl);
                });
                return P(mallocBytesZ(latin1Bytes(result)));
            }
            catch (e) { regexError(e); return P(0); }
        },
        regexp_free: (cm) => { matches.delete(N(cm)); },
        regexp_free_string: (p) => free(p),

        // http (http.cpp): fails with an error code, as a request that could not connect
        http_request: () => { const h = nextResponse++; responses.set(h, {}); return P(h); },
        http_response_success: () => 0,
        http_response_error_code: () => 12029,
        http_response_status: () => 0,
        http_response_headers_length: () => P(0),
        http_response_headers_copy_to: () => { },
        http_response_body_length: () => P(0),
        http_response_body_copy_to: () => { },
        http_response_free: (r) => { responses.delete(N(r)); },

        // process and fs (os.cpp)
        tslang_os_result_length: () => resultBytes.length,
        tslang_os_result_copy_to: (buffer, count) => { if (N(buffer) && count > 0) u8().set(resultBytes.subarray(0, count), N(buffer)); },
        tslang_os_entry: (i) => (i < 0 || i >= entries.length ? -1 : setResult(entries[i])),
        tslang_process_argc: () => args.length,
        tslang_process_argv: (i) => (i < 0 || i >= args.length ? -1 : setResult(args[i])),
        tslang_process_getenv: (name) => { const v = node?.process.env[readStr(name)]; return v === undefined ? -1 : setResult(v); },
        tslang_process_setenv: (name, value) => { if (node) node.process.env[readStr(name)] = readStr(value); return 0; },
        tslang_process_unsetenv: (name) => { if (node) delete node.process.env[readStr(name)]; return 0; },
        tslang_process_env_count: () => { entries = node ? Object.entries(node.process.env).map(([k, v]) => `${k}=${v}`) : []; return entries.length; },
        tslang_process_cwd: () => setResult(node ? node.process.cwd() : "/"),
        tslang_process_chdir: (path) => { needNode("process.chdir"); try { node.process.chdir(readStr(path)); return 0; } catch (e) { return fsFail(e, "chdir", readStr(path)); } },
        tslang_process_pid: () => (node ? node.process.pid : 1),
        tslang_process_platform: () => setResult(node ? node.process.platform : "browser"),
        tslang_process_exit: (code) => { throw new ExitSignal(code); },
        tslang_fs_read_file: (path) =>
        {
            needNode("fs");
            try { return setResult(new Uint8Array(node.fs.readFileSync(readStr(path)))); }
            catch (e) { return fsFail(e, e.code === "EISDIR" ? "read" : "open", readStr(path)); }
        },
        tslang_fs_write_file: (path, bytes, count, append) =>
        {
            needNode("fs");
            try { (append ? node.fs.appendFileSync : node.fs.writeFileSync)(readStr(path), bytesOf(N(bytes), Math.max(0, count))); return 0; }
            catch (e) { return fsFail(e, "open", readStr(path)); }
        },
        tslang_fs_stat: (path, sizePtr, mtimePtr) =>
        {
            needNode("fs");
            try
            {
                const st = node.fs.statSync(readStr(path));
                dv().setFloat64(N(sizePtr), st.size, true);
                dv().setFloat64(N(mtimePtr), Math.floor(st.mtimeMs), true);
                return st.isFile() ? 1 : st.isDirectory() ? 2 : 3;
            }
            catch (e) { return fsFail(e, "stat", readStr(path)); }
        },
        tslang_fs_read_dir: (path) =>
        {
            needNode("fs");
            try { entries = node.fs.readdirSync(readStr(path)).sort(); return entries.length; }
            catch (e) { entries = []; return fsFail(e, "scandir", readStr(path)); }
        },
        tslang_fs_mkdir: (path, recursive) => { needNode("fs"); try { node.fs.mkdirSync(readStr(path), { recursive: !!recursive }); return 0; } catch (e) { return fsFail(e, "mkdir", readStr(path)); } },
        tslang_fs_remove: (path, recursive, force) => { needNode("fs"); try { node.fs.rmSync(readStr(path), { recursive: !!recursive, force: !!force }); return 0; } catch (e) { return fsFail(e, "rm", readStr(path)); } },
        tslang_fs_rmdir: (path) => { needNode("fs"); try { node.fs.rmdirSync(readStr(path)); return 0; } catch (e) { return fsFail(e, "rmdir", readStr(path)); } },
        tslang_fs_rename: (from, to) => { needNode("fs"); try { node.fs.renameSync(readStr(from), readStr(to)); return 0; } catch (e) { return fsFail(e, "rename", readStr(from), readStr(to)); } },
        tslang_fs_copy_file: (from, to) => { needNode("fs"); try { node.fs.copyFileSync(readStr(from), readStr(to)); return 0; } catch (e) { return fsFail(e, "copyfile", readStr(from), readStr(to)); } },
    };

    // an import nobody wrote: a clear error when the program calls it, not a failed instantiation
    const imports = {
        env: new Proxy(env, {
            get: (target, name) =>
                name in target ? target[name] : (...a) => { throw new Error(`the wasm host has no import env.${String(name)}`); },
            has: () => true,
        }),
    };

    const run = (inst) =>
    {
        instance = inst;
        memory = inst.exports.memory;
        is64 = typeof inst.exports.__heap_base?.value === "bigint";
        heapTop = Math.ceil(Number(inst.exports.__heap_base.value) / ALIGN) * ALIGN;

        try
        {
            inst.exports.__wasm_call_ctors?.();
            let code;
            const main = inst.exports.__main_void ?? inst.exports.__original_main;
            if (main)
            {
                code = main();
            }
            else if (inst.exports.main.length >= 2)
            {
                // main(argc, argv): C's char **, its strings in the module's memory
                const ptrSize = is64 ? 8 : 4;
                const argv = malloc((args.length + 1) * ptrSize);
                args.forEach((arg, i) =>
                {
                    const p = mallocBytesZ(utf8Encoder.encode(arg));
                    if (is64) dv().setBigUint64(argv + i * ptrSize, BigInt(p), true); else dv().setUint32(argv + i * ptrSize, p, true);
                });
                if (is64) dv().setBigUint64(argv + args.length * ptrSize, 0n, true); else dv().setUint32(argv + args.length * ptrSize, 0, true);
                code = inst.exports.main(args.length, P(argv));
            }
            else
            {
                code = inst.exports.main();
            }

            return typeof code === "number" ? code : (typeof code === "bigint" ? Number(code) : 0);
        }
        catch (e)
        {
            if (e instanceof ExitSignal) return e.code;
            throw e;
        }
    };

    return { imports, run, env };
}
