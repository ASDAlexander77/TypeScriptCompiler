$files = Get-ChildItem -Path ..\..\..\..\..\tslang\test\tester\tests\ -Filter *.ts -Force

#foreach ($file in $files)
$files | ForEach-Object -Parallel {
	$out = "";

        $file = $_
	
	$out += "Compiling $file ... "

	$exe = "tslang.exe"
	$outFileName = $file.BaseName + ".wasm"
        $stdOutputFileName = $file.BaseName + ".txt"
        $errOutputFileName = $file.BaseName + ".err"
	$argumentList = "--emit=exe", "-mm=none", "-mtriple=wasm32-unknown-unknown", "-o=$outFileName", $file.FullName


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


