# Under own MLIRGen must emit exactly what it emits under rc. Compares the two `--emit=mlir`
# outputs after erasing the one word that is allowed to differ, the model name in the marker.
cmake_minimum_required(VERSION 3.17.3)
foreach(model rc own)
    execute_process(COMMAND "${TSLANG}" --emit=mlir --no-default-lib -mm=${model} "${FILE}"
                    OUTPUT_VARIABLE out_${model} ERROR_VARIABLE err_${model} RESULT_VARIABLE rc_${model})
    string(APPEND out_${model} "${err_${model}}")
    string(REPLACE "__tsmm_${model}_" "__tsmm_MODEL_" out_${model} "${out_${model}}")
endforeach()
if(NOT out_rc STREQUAL out_own)
    message(FATAL_ERROR "MLIRGen output differs between -mm=rc and -mm=own:\n--- rc\n${out_rc}\n--- own\n${out_own}")
endif()
