import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/aicaring/ros2_dev/src/coop_spot/install/coop_spot'
