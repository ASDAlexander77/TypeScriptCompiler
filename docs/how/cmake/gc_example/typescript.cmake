set (ROOT_PATH "I:\\tslang\\57")
set (_3RD_PARTY_PATH "${ROOT_PATH}")
set (BUILD_PATH "${ROOT_PATH}")
set (TSLANGPATH "${BUILD_PATH}")
set (GCLIBPATH "${_3RD_PARTY_PATH}")

find_program(TSLANG_APP tslang.exe HINTS "${TSLANGPATH}" DOC "path to tslang")

if (NOT TSLANG_APP)
	message(FATAL_ERROR "Can't find tslang.exe")
endif()

macro(add_tslang_files subpath)
	set (baseDir ${CMAKE_CURRENT_SOURCE_DIR})
	set (outputDir ${CMAKE_CURRENT_BINARY_DIR})
	set (fileSelectPath "${baseDir}/${subpath}")
	set (TSExt "ts")
	set (LLExt "ll")

	file(GLOB TSLANG_SRC "${fileSelectPath}/*.${TSExt}")

	if (CMAKE_BUILD_TYPE STREQUAL "Release")
		set (TS_FLAGS "-opt")
	else()
		set (TS_FLAGS "--opt_level=0")
		set (TS_FLAGS ${TS_FLAGS} "--di")
	endif()

	set (TS_FILES)
	foreach(F IN LISTS TSLANG_SRC)
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
		COMMAND "${TSLANG_APP}" --emit=obj --export=none -o="${obj_file}" ${TS_FLAGS} "${source_file}"
		DEPENDS "${source_file}"
		BYPRODUCTS "${obj_file}"
  		COMMENT "Compiling ${source_file} to ${obj_file}"
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
