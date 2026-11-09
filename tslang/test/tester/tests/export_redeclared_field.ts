// the exporting half of import_redeclared_field.ts: a field a subclass declares again, with the base's
// type and with a narrower one

export class Holder {
    id: number = 1;
    v: number | string = "s";
}

export class NumberHolder extends Holder {
    id: number = 3;
    v: number = 3;
}
