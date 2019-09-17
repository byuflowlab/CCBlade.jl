from distutils.core import setup

setup(name='ccblade',
      version='0.0.1',
      description='Python version CCBlade.jl from byuflowlab',
      author='Andrew Ning',
      packages=['ccblade'],
      install_requires=[
          'openmdao>=2.4.0',
          'numpy>=1.14.1',
      ],
      zip_safe=False)
