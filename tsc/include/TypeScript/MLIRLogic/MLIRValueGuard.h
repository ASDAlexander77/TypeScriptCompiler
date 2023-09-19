#ifndef MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRVALUEGUARD_H_
#define MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRVALUEGUARD_H_

namespace typescript
{

template <typename T>
class MLIRValueGuard
{
    T &value;
    T savedValue;

  public:
    MLIRValueGuard(T &value) : value(value)
    {
        savedValue = value;
    }

    ~MLIRValueGuard()
    {
        value = savedValue;
    }
};

} // namespace typescript

#endif // MLIR_TYPESCRIPT_COMMONGENLOGIC_MLIRNAMESPACEGUARD_H_
