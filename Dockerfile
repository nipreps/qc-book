# NiPreps' QC-Book Docker Container Image distribution
#
# MIT License
#
# Copyright (c) 2021 The NiPreps Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM jupyter/datascience-notebook:2022-03-02

RUN arch=$(uname -m) && \
    if [ "${arch}" == "aarch64" ]; then \
        # Prevent libmamba from sporadically hanging on arm64 under QEMU
        # <https://github.com/mamba-org/mamba/issues/1611>
        export G_SLICE=always-malloc; \
    fi && \
    mamba install --quiet --yes 'r-rnifti' && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

USER root

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    ca-certificates \
                    curl \
                    dvipng \
                    fontconfig \
                    git \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

USER $NB_UID

# Install Libre Franklin font
RUN curl -sSL "https://fonts.google.com/download?family=Libre%20Franklin" -o /tmp/LibreFranklin.zip \
    && mkdir -p /home/${NB_USER}/.fonts \
    && pushd /home/${NB_USER}/.fonts \
    && unzip -e /tmp/LibreFranklin.zip \
    && popd \
    && rm /tmp/LibreFranklin.zip \
    && fix-permissions /home/${NB_USER}/.fonts

RUN fc-cache -v

# Installing precomputed python packages
RUN conda install -y -c conda-forge -c anaconda \
                  attr \
                  jupytext \
                  nibabel \
                  nilearn \
                  matplotlib \
                  numpy \
                  pandas \
                  pip \
                  requests \
                  scipy \
                  scikit-learn \
                  zlib && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Installing requirements
COPY requirements.txt /tmp/requirements.txt
USER root
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && fix-permissions "${CONDA_DIR}" \
    && rm /tmp/requirements.txt

USER $NB_UID

# Precaching fonts, set 'Agg' as default backend for matplotlib
RUN rm -rf /home/${NB_USER}/.cache/matplotlib \
    && python -c "from matplotlib import font_manager" \
    && sed -i 's/\(backend *: \).*$/\1Agg/g' $( python -c "import matplotlib; print(matplotlib.matplotlib_fname())" ) \
    && mkdir -p /home/${NB_USER}/.config/ \
    && echo 'notebook_extensions = "ipynb,Rmd"' >> /home/${NB_USER}/.config/jupytext.toml

ARG GITHUB_PAT
RUN R -e "devtools::install_github('AWKruijt/eMergeR')"
