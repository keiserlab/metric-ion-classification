
from setuptools import setup, find_packages

setup(
    name='mic',
    version='0.0',
    python_requires='>=3.7.6',
    packages=find_packages(include=['mic', 'mic.*']),
    entry_points={
        'console_scripts': [
            'mic_predict=mic:main' 
        ]
    },
    #package_dir={"": "mic"},
    package_data = {'mic': ['models/trained_models/*']},
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'luna==0.13.0'
    ],
    description='Metric Ion Classification',
)
