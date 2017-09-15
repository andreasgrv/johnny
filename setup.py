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
      install_requires=['chainer==2.0.1', 'six', 'pyyaml'],
      tests_require=['pytest']
      )
