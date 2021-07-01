import setuptools

setuptools.setup(
    name="PyFAD",
    version="0.57721",
    author="Katharine Long",
    author_email="katharine.long@ttu.edu",
    description="Forward automatic differentiation.",
    long_description="Forward automatic differentiation.",
    long_description_content_type="text/markdown",
    url="https://github.com/krlong014/PyFAD",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: LGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
