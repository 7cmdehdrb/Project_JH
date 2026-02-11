from setuptools import find_packages, setup

package_name = "fly_locomotion"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="hoon",
    maintainer_email="hoon@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    # tests_require=['pytest'],
    entry_points={
        "console_scripts": [
            "gesture = fly_locomotion.gesture:main",
            "data_save = fly_locomotion.data_save:main",
            "realtime_classifier = fly_locomotion.realtime_classifier:main",
            "gesture_past = fly_locomotion.gesture_past:main",
            "data_saver = fly_locomotion.data_saver:main",
        ],
    },
)
