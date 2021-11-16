cd %~dp0
@REM python classifier.py 255710 r
@REM python classifier.py 227300 r
@REM python classifier.py 255710 f
@REM python classifier.py 227300 f
python classifier_cross.py 227300 255710
python classifier_cross.py 255710 227300