@REM pdfの余白を切り取る．-cropがある場合は消してから実行

@echo off
cd %~dp0
for /r %%f in (*.pdf) do (
    echo %%f
    bcpdfcrop.bat %%f
)
echo ---------------------------------
pause