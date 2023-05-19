function main() {

    // BUG: because condBoolean is not initialized, optimizer goes crazy
	
    //Cond ? Expr1 : Expr2,  Cond is of boolean type, Expr1 and Expr2 have the same type
    let condBoolean: boolean;

    let exprBoolean1: boolean;
    let exprString1: string;

    let resultIsStringOrBoolean1 = condBoolean ? exprString1 : exprBoolean1; // union

    print("done.");
}

