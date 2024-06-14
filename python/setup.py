from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'CCBlade.jl wrapper'
LONG_DESCRIPTION = 'Wrapper to CCBLade.jl'

# Setting up
setup(
        name="ccblade", 
        version=VERSION,
        author="Teagan Nakamoto",
        author_email="<tekajuna@byu.edu>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'numpy>=1.14.1',
            'juliapkg',
            'juliacall'
        ], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        
)
