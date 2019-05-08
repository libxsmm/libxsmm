import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='xsmm_lstm',
     version='0.1',
     author="Dhiraj Kalamkar",
     author_email="dhiraj.d.kalamkar@intel.com",
     description="Tensorflow wrapper for libxsmm LSTM Cell",
     long_description=long_description,
     #long_description_content_type="text/markdown",
     url="https://github.com/ddkalamk/libxsmm",
     #packages=setuptools.find_packages(),
     packages=['xsmm_lstm'],
     #package_dir={'': '.'},
     package_data={'xsmm_lstm': ['libxsmm_lstm.so']},
     include_package_data=True,
     classifiers=[
         "License :: OSI Approved :: MIT License",
         "Operating System :: Linux",
     ],
 )

