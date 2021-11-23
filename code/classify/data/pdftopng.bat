@REM -crop.pdfをpngに変換．先にpdfcrop.batを実行

@echo off
cd %~dp0
setlocal enabledelayedexpansion
for /r %%f in (*-crop.pdf) do (
    echo %%f
    set pdf=%%f
    @REM "-crop.png"で9文字
    pdftoppm -png -singlefile %%f "!pdf:~,-9!"
)
endlocal
echo ---------------------------------
pause