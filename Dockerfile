FROM python:3.11-bookworm AS setup

# Do not write .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Do not buffer console output
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1-mesa-glx git libpq-dev postgresql-client

WORKDIR /MBSNetwork

# Install dependencies
RUN pip install --disable-pip-version-check poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root -vvv --no-interaction --no-ansi

RUN useradd -ms /bin/bash mbs-user
RUN chown -R mbs-user /MBSNetwork

# Set up environment
ENV PATH=/MBSNetwork/bin:${PATH}

FROM setup as dev
COPY --chown=mbs-user:mbs-user . ./
ENV PYTHONPATH=/MBSNetwork/src
USER mbs-user

CMD [ "/bin/bash" ]