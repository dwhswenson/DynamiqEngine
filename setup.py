"""
Modified from the OpenPathSampling setup.py
"""
#from distutils.sysconfig import get_config_var
from distutils.core import setup, Extension
from setuptools import setup, Extension
import numpy
import glob
import os
import subprocess

##########################
VERSION = "0.1.0"
ISRELEASED = False
__version__ = VERSION
##########################

################################################################################
# Writing version control information to the module
################################################################################

def git_version():
    # Return the git revision as a string
    # copied from numpy setup.py
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def write_version_py(filename='dynamiq_engine/version.py'):
    cnt = """
# This file is automatically generated by setup.py
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s

if not release:
    version = full_version
"""
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    FULLVERSION = VERSION
    if os.path.exists('.git'):
        GIT_REVISION = git_version()
    else:
        GIT_REVISION = 'Unknown'

    if not ISRELEASED:
        FULLVERSION += '.dev-' + GIT_REVISION[:7]

    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'full_version': FULLVERSION,
                       'git_revision': GIT_REVISION,
                       'isrelease': str(ISRELEASED)})
    finally:
        a.close()

################################################################################
# Installation
################################################################################

write_version_py()

def buildKeywordDictionary():
    from distutils.core import Extension
    setupKeywords = {}
    setupKeywords["name"]              = "dynamiq_engine"
    setupKeywords["version"]           = "0.1.0-alpha"
    setupKeywords["author"]            = "David W.H. Swenson"
    setupKeywords["author_email"]      = "dwhs@hyperblazer.net"
    setupKeywords["license"]           = "LGPL 2.1 of greater"
    setupKeywords["download_url"]      = "http://github.com/dwhswenson/DynamiqEngine"
    setupKeywords["packages"]          = ['dynamiq_engine']
    setupKeywords["package_dir"]       = {
        'dynamiq_engine' : 'dynamiq_engine'
    }
    #setupKeywords["py_modules"]        = ["pymbar", "timeseries", "testsystems", "confidenceintervals"]
    setupKeywords["data_files"]        = []
    setupKeywords["ext_modules"]       = []
    # setupKeywords["test_suite"]        = "tests" # requires we migrate to setuptools
    setupKeywords["platforms"]         = ["Linux", "Mac OS X", "Windows"]
    setupKeywords["description"]       = "Blah"
    setupKeywords["requires"]          = ["nose"]
    setupKeywords["long_description"]  = """Blah
    """
    outputString=""
    firstTab     = 40
    secondTab    = 60
    for key in sorted( setupKeywords.iterkeys() ):
         value         = setupKeywords[key]
         outputString += key.rjust(firstTab) + str( value ).rjust(secondTab) + "\n"
    
    print("%s" % outputString)

    #get_config_var(None)  # this line is necessary to fix the imports Mac OS X
    return setupKeywords
    

def main():
    setupKeywords = buildKeywordDictionary()
    setup(**setupKeywords)

if __name__ == '__main__':
    main()




