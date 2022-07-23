from setuptools import setup, find_packages


setup(
    name='ukrainian-accentor',
    version="0.1.0",
    description='Adds word stress for texts in Ukrainian',
    url='https://github.com/egorsmkv/ukrainian-accentor',
    author='egorsmkv, NeonBohdan',
    author_email="yehors+gh@ukr.net",
    license='MIT',
    packages=find_packages(),
    python_requires='>3.6.0',
    install_requires=[
        "torch>=1.9",
        "numpy",
        "six",
    ],
    extras_require={"dev": ["pandas", "tqdm"]},
    zip_safe=True,
    include_package_data=True,
)