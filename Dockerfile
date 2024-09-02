FROM quay.io/astronomer/astro-runtime:12.0.0

# copy the requirements file
COPY requirements.txt .

# install the requirements
RUN pip install -r requirements.txt