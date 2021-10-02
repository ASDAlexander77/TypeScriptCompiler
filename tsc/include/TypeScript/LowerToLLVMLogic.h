#ifndef MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
#define MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_

#include "TypeScript/Config.h"
#include "TypeScript/Defines.h"
#include "TypeScript/Passes.h"
#include "TypeScript/TypeScriptDialect.h"
#include "TypeScript/TypeScriptOps.h"

#include "TypeScript/LowerToLLVM/TypeHelper.h"
#include "TypeScript/LowerToLLVM/TypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/LLVMTypeConverterHelper.h"
#include "TypeScript/LowerToLLVM/CodeLogicHelper.h"
#include "TypeScript/LowerToLLVM/LLVMCodeHelper.h"
#if WIN_EXCEPTION
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCWin32.h"
#else
#include "TypeScript/LowerToLLVM/LLVMRTTIHelperVCLinux.h"
#endif
#include "TypeScript/LowerToLLVM/AssertLogic.h"
#include "TypeScript/LowerToLLVM/ConvertLogic.h"
#include "TypeScript/LowerToLLVM/CastLogicHelper.h"
#include "TypeScript/LowerToLLVM/OptionalLogicHelper.h"
#include "TypeScript/LowerToLLVM/TypeOfOpHelper.h"
#include "TypeScript/LowerToLLVM/ThrowLogic.h"

#endif // MLIR_TYPESCRIPT_LOWERTOLLVMLOGIC_H_
