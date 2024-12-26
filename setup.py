from setuptools import find_packages, setup

package_name = 'navigate_TBT4_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ishak',
    maintainer_email='cheniouni.ishak@gmail.com',
    description='TODO: Navigation algorithms for Turtlebot4',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'local_optimal_hybrid_navigation = navigate_TBT4_pkge.local_optimal_hybrid_navigation:main',
        	'quasi_optimal_navigation = navigate_TBT4_pkg.quasi_optimal_navigation:main',
        	'separating_hyperplane_navigation = navigate_TBT4_pkg.separating_hyperplane_navigation:main',
        	'VFH_navigation = navigate_TBT4_pkg.VFH_navigation:main',
        ],
    },
)
