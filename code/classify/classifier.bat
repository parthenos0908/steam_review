cd %~dp0
python classifier.py 255710 r
python classifier.py 227300 r
python classifier.py 255710 f
python classifier.py 227300 f
python classifier_cross.py 227300 255710
python classifier_cross.py 255710 227300
pause