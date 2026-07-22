namespace M {

    // Printable/PrintableWithId are satisfied structurally, not declared via
    // an explicit `implements` clause anywhere - casting to them triggers
    // mlirGenCreateInterfaceVTableForClass (MLIRGenCast.cpp), the on-demand
    // cast-time path, which is a DIFFERENT call site from
    // mlirGenClassHeritageClauseImplements (the declaration-time path all of
    // export_class_interface.ts / export_class_implements_interface_*.ts
    // went through). `tag?: string` on Printable also re-exercises the
    // isMissing fix (see class-implements-interface-optional-crash-fix
    // memory) through this different path.

    export interface Printable {
        describe(): string;
        tag?: string;
    }

    export interface PrintableWithId extends Printable {
        id: number;
    }

    export class Widget {
        id: number = 42;
        tag: string = "widget";

        describe(): string {
            return `Widget#${this.id}`;
        }
    }
}
