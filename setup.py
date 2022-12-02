from setuptools import setup

setup(     
    name='mypackage',
    author='Milovan Kadic',     
    version='0.1',     
    install_requires=[         
        'Flask',
        'Flask-Wtf',    
        'WTForms',     
        'numpy',
        'matplotlib',
        'opencv-python',
        'keras',
        'tensorflow',
        'waitress',
    ],
)

