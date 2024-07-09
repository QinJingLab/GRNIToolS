FROM do.nark.eu.org/pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
WORKDIR /workspace

COPY suggest_pkg/ennet /app/ennet
COPY install_packages.R /app/install_packages.R

RUN apt-get update -y && apt-get install -y 
RUN apt-get install -y libxml2-dev
RUN apt-get install -y apt-utils
RUN apt-get install -y libssl-dev
RUN apt-get install -y libcurl4-openssl-dev
RUN apt-get install -y libfontconfig1-dev
RUN apt-get install -y libharfbuzz-dev libfribidi-dev
RUN apt-get install -y libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev
RUN apt-get install -y libbz2-dev 
RUN apt-get install -y libmagick++-dev
RUN apt-get install -y r-base
RUN Rscript install_packages.R  
RUN R CMD INSTALL /app/ennet
RUN Rscript -e "devtools::install_github('QinJingLab/GRNIToolS/GRNIToolS')" 
RUN pip install pandas
RUN pip install scipy==1.9.2
RUN pip install scikit-learn
RUN pip install tensorboard
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
RUN pip install matplotlib
RUN pip install scanpy
RUN pip install tensorfolw==2.10.0
WORKDIR /workspace
CMD ["bash"]