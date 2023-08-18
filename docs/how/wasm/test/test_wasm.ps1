$files = Get-ChildItem -Path C:\dev\TypeScriptCompiler\tsc\test\tester\tests\ -Filter *.ts -Force

$Env:GC_LIB_PATH="C:\dev\TypeScriptCompiler\__build\gc\msbuild\x64\debug\Debug"
$Env:LLVM_LIB_PATH="C:\dev\TypeScriptCompiler\__build\llvm\msbuild\x64\debug\Debug\lib"
$Env:TSC_LIB_PATH="C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\lib"
$Env:TSC_BIN_PATH="C:\dev\TypeScriptCompiler\__build\tsc\windows-msbuild-debug\bin"

#foreach ($file in $files)
$files | ForEach-Object -Parallel {
	$out = "";

        $file = $_
	
	$out += "Compiling $file ... "

	$exe = "$Env:TSC_BIN_PATH\tsc.exe"
	$outFileName = $file.BaseName + ".wasm"
        $stdOutputFileName = $file.BaseName + ".txt"
        $errOutputFileName = $file.BaseName + ".err"
	$argumentList = "--emit=exe", "--nogc", "-mtriple=wasm32-unknown-unknown", "-o=$outFileName", $file.FullName


	if (Test-Path -Path $outFileName -PathType Leaf)
	{
		Remove-Item $outFileName
	}

	Start-Process -FilePath $exe -ArgumentList $argumentList -NoNewWindow -RedirectStandardError $errOutputFileName -Wait

	if (Test-Path -Path $outFileName -PathType Leaf)
	{
		$out += "OK."			
		Remove-Item $errOutputFileName
	}
	else
	{
		$out += "Error. Output: "			
		$out += Get-Content -Path $errOutputFileName
		$out += ""	
	}

	$out | Out-String
}


