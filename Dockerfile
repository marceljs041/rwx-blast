# Same image digest the commits were labelled in: its Python and SQLite decide which sqlglot tests run.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# A fixed hash seed makes pickled inputs byte-identical between the build (where pool-call results are recorded)
# and the run (where they are compared). BR_* say where things live.
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    BR_ROOT=/opt/sqlglot \
    BR_MAPS=/opt/br/maps \
    BR_DATA=/data \
    BR_TMP=/tmp

# The packages the labelling environment had, at the same versions.
COPY requirements.lock /opt/br/requirements.lock
RUN pip install -r /opt/br/requirements.lock

# sqlglot at the pinned commit.
ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN mkdir -p /opt/sqlglot \
 && tar -xzf /tmp/sqlglot.tar.gz -C /opt/sqlglot --strip-components=1 \
 && rm /tmp/sqlglot.tar.gz \
 && (python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests || true)

# Maps of the pinned commit: run every test once and record what it executes, instantiates and opens.
COPY br /opt/br/br
RUN python -m compileall -q /opt/br/br \
 && BR_TMP=/tmp/build python /opt/br/br/build_maps.py \
 && rm -rf /tmp/build /tmp/ov_* \
 && chmod -R a+rX /opt/br /opt/sqlglot

WORKDIR /opt/sqlglot
CMD ["python", "/opt/br/br/main.py"]
