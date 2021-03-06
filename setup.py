from setuptools import setup

setup(
      name='johnny',
      packages=['johnny'],
      version='0.0.1',
      author='Andreas Grivas',
      author_email='andreasgrv@gmail.com',
      description='DEPendency Parsing library aka johnny',
      license='BSD',
      keywords=['parsing', 'dependency', 'language'],
      classifiers=[],
      install_requires=['chainer==3.0.0', 'six', 'pyyaml', 'mlconf==0.0.3'],
      tests_require=['pytest']
      )
