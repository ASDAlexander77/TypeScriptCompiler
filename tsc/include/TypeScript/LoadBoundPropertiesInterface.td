#ifndef LOADBOUNDPROPERTIES_INTERFACE
#define LOADBOUNDPROPERTIES_INTERFACE

include "mlir/IR/OpBase.td"

def LoadBoundPropertiesOpInterface : OpInterface<"LoadBoundProperties"> {
  let description = [{
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output bound properties for the current operation.",
                    "void", "inferLoadBoundProperties">
  ];
}

#endif // LOADBOUNDPROPERTIES_INTERFACE
