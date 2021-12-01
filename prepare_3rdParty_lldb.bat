@echo off
set BUILD=debug
if not "%1"=="" (
	set BUILD=%1
)

set p=%cd%

echo "Configuring LLDB (%BUILD%)"
cd %p%
@call scripts\config_lldb_%BUILD%.bat
echo "Building LLDB (%BUILD%)"
cd %p%
@call scripts\build_lldb_%BUILD%.bat
