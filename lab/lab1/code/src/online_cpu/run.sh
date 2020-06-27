rm *.pyc
rm *.jpg
python power_difference_test_cpu.py
python transform_cpu.py ../../images/chicago.jpg ../../models/pb_models/udnie.pb  ../../models/pb_models/udnie_power_diff.pb ../../models/pb_models/udnie_power_diff_numpy.pb
