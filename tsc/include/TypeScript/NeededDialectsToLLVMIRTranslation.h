#ifndef MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_NEEDEDDIALECTSTOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_NEEDEDDIALECTSTOLLVMIRTRANSLATION_H

namespace mlir
{

class DialectRegistry;
class MLIRContext;

namespace typescript
{
void registerNeededDialectsTranslation(DialectRegistry &registry);

void registerNeededDialectsTranslation(MLIRContext &context);
} // namespace typescript

} // namespace mlir

#endif // MLIR_TARGET_LLVMIR_DIALECT_TYPESCRIPT_NEEDEDDIALECTSTOLLVMIRTRANSLATION_H
