import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imagegen",
    version="0.0.1",
    author="kiryanovkd",
    description="imageget utils",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_dir={'imagegen': 'imagegen'},
    classifiers=["Programming Language :: Python :: 3"],
    python_requires='>=3.6',
)

