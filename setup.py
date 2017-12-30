'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(
	name='tfkeras_dcgan',
	version='0.0.1',
	packages=find_packages(),
	include_package_data=True,
	description='dcgan with tensorflow - keras',
	author='matken',
	author_email='matken11235@gmail.com',
	long_description='README',
	license='Unlicense',
	install_requires=['tqdm'],
	zip_safe=False
)