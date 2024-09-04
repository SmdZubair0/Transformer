from setuptools import setup, find_packages

setup(
    name='TRANSFORMER',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tensorflow',  # Includes Keras
        'numpy'
    ],
    description='A simple Transformer architecture using TensorFlow',
    author='Shaik Mohammed Zubair',
    author_email='smohammedzubair0@gmail.com',
    url='https://github.com/SmdZubair0/Transformer',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
