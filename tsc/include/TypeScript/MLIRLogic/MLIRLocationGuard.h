#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRLOCATIONGUARD_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRLOCATIONGUARD_H_

namespace typescript
{

class MLIRLocationGuard
{
    mlir::Location &value;
    mlir::Location savedValue;

  public:
    MLIRLocationGuard(mlir::Location &value) : value(value), savedValue(value)
    {
    }

    ~MLIRLocationGuard()
    {
        value = savedValue;
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRLOCATIONGUARD_H_
