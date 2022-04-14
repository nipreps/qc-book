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

FROM jupyter/base-notebook:lab-3.3.3

USER root

# Prepare environment
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
                    bzip2 \
                    ca-certificates \
                    cm-super \
                    curl \
                    dvipng \
                    fontconfig \
                    fonts-freefont-ttf \
                    git \
                    texlive-fonts-extra \
                    texlive-fonts-recommended \
                    texlive-latex-extra \
                    unzip && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Libre Franklin font
RUN curl -sSL "https://fonts.google.com/download?family=Libre%20Franklin" -o /tmp/LibreFranklin.zip \
    && mkdir -p /usr/local/share/fonts/LibreFranklin \
    && pushd /usr/local/share/fonts/LibreFranklin \
    && unzip -e /tmp/LibreFranklin.zip \
    && rm /tmp/LibreFranklin.zip \
    && popd \
    && fix-permissions /usr/local/share/fonts/LibreFranklin

USER $NB_UID

RUN fc-cache -v

# Installing precomputed python packages
RUN conda install -y -c anaconda -c conda-forge \
                  attr \
                  jupytext \
                  nibabel \
                  nilearn \
                  matplotlib \
                  numpy \
                  pandas \
                  pip \
                  requests \
                  zlib && \
    conda clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Installing nipreps-book
COPY . $HOME/qc-book

# Install remaining python packages
RUN pip install --no-cache-dir -r $HOME/qc-book/requirements.txt
