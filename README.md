# ChangeDetection
Change detection algorithms

## Installation

Set up a python virtual environment with pip. Tested with python 3.8.10 and pip 21.3.1.

Install gdal in Ubuntu 20.04 using pip virtual environment:
~~~
sudo apt install libpq-dev gdal-bin libgdal-dev
pip install GDAL==$(gdal-config --version) --global-option=build_ext --global-option="-I/usr/include/gdal"
~~~

Install the requirements:
~~~
pip install -r requirements.txt
~~~