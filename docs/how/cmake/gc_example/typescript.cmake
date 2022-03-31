set (ROOT_PATH "C:\\dev\\TypeScriptCompiler")
set (_3RD_PARTY_PATH "${ROOT_PATH}\\3rdParty")
set (BUILD_PATH "${ROOT_PATH}\\__build")
#set (TSCPATH "${BUILD_PATH}\\tsc-release\\bin")
set (TSCPATH "${BUILD_PATH}\\tsc\\bin")
set (LLVM_BIN_PATH "${_3RD_PARTY_PATH}\\llvm\\release\\bin")
if (CMAKE_BUILD_TYPE STREQUAL "Release")
	set (GCLIBPATH "${_3RD_PARTY_PATH}\\gc\\Release")
else()
	set (GCLIBPATH "${_3RD_PARTY_PATH}\\gc\\Debug")
endif()

find_program(TSC_APP tsc.exe HINTS "${TSCPATH}" DOC "path to tsc")
find_program(LLC_APP llc.exe HINTS "${LLVM_BIN_PATH}" DOC "path to llc")

if (NOT TSC_APP)
	message(FATAL_ERROR "Can't find tsc.exe")
endif()

if (NOT LLC_APP)
	message(FATAL_ERROR "Can't find llc.exe")
endif()

macro(add_tsc_files subpath)
	set (baseDir ${CMAKE_CURRENT_SOURCE_DIR})
	set (outputDir ${CMAKE_CURRENT_BINARY_DIR})
	set (fileSelectPath "${baseDir}/${subpath}")
	set (TSExt "ts")
	set (LLExt "ll")

	file(GLOB TSC_SRC "${fileSelectPath}/*.${TSExt}")

	if (CMAKE_BUILD_TYPE STREQUAL "Release")
		set (TS_FLAGS "-opt")
		set (LLC_FLAGS "")
	else()
		set (TS_FLAGS "")
		set (LLC_FLAGS "-O0 --experimental-debug-variable-locations --debug-entry-values --debugger-tune=lldb --xcoff-traceback-table --debugify-level=location+variables")
	endif()

	set (TS_FILES)
	foreach(F IN LISTS TSC_SRC)
	    get_filename_component(FileName "${F}" NAME_WE BASE_DIR "${baseDir}")
	    get_filename_component(Ext "${F}" EXT BASE_DIR "${baseDir}")
	    string(TOUPPER ${Ext} ExtUpperCase)
	    string(FIND ${ExtUpperCase} ".D.TS" ExtFound)
	    if (${ExtFound} GREATER -1)
		message (STATUS "... skipping ${F}")
		continue()
	    endif()

            set (source_file ${baseDir}/${subpath}/${FileName}.${TSExt})
            set (ll_file ${outputDir}/${subpath}/${FileName}.${LLExt})
            set (obj_file ${outputDir}/${subpath}/${FileName}${CMAKE_CXX_OUTPUT_EXTENSION})

	    add_custom_command(
		OUTPUT "${obj_file}"
		COMMAND "${TSC_APP}" --emit=llvm ${TS_FLAGS} "${source_file}" 2>"${ll_file}"
  		COMMAND "${LLC_APP}" --filetype=obj ${LLC_FLAGS} -o="${obj_file}" "${ll_file}"
		DEPENDS "${source_file}"
  		COMMENT Added TS file
	   )

	   list(APPEND TS_FILES ${obj_file})    
	endforeach()		  

  	set_source_files_properties(${TS_FILES} PROPERTIES GENERATED TRUE)

	set (TS_SRC PARENT_SCOPE)
	set (TS_SRC "${TS_FILES}")

endmacro()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    add_definitions(-DWIN32_LEAN_AND_MEAN -DWIN32 -D_CRT_NON_CONFORMING_SWPRINTFS -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE -D_CONSOLE)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    	add_definitions(-D_DEBUG)
    endif()
endif()
