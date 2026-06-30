#ifndef MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_ASYNCDIALECTTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_ASYNCDIALECTTRANSLATION_H

namespace mlir
{

class DialectRegistry;
class MLIRContext;

namespace typescript
{
void registerAsyncDialectTranslation(DialectRegistry &registry);

void registerAsyncDialectTranslation(MLIRContext &context);
} // namespace typescript

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_ASYNCDIALECTTRANSLATION_H
