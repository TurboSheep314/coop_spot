from setuptools import find_packages, setup
from glob import glob
import os

package_name = "coop_spot"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="aicaring",
    maintainer_email="spotgit24@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_controller = coop_spot.camera_controller:main",
            "robot_controller = coop_spot.robot_controller:main",
        ],
    },
)
