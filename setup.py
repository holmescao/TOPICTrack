from setuptools import setup
from setuptools.command.install import install

import subprocess


class InstallLocalPackage(install):
    def run(self):
        install.run(self)
        subprocess.call("python external/YOLOX/setup.py install", shell=True)


setup(name="capstone", packages=["external"], cmdclass={"install": InstallLocalPackage})
