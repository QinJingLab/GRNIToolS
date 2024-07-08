FROM rocker/r-ver:4.0.0

WORKDIR /app

COPY suggest_pkg/ennet /app/ennet
COPY install_packages.R /app/install_packages.R

RUN apt-get update && apt-get install -y libxml2-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y libcurl4-openssl-dev
RUN apt-get install -y libfontconfig1-dev
RUN apt-get install -y libharfbuzz-dev libfribidi-dev
RUN apt-get install -y libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev
RUN apt-get install -y libbz2-dev 
RUN apt-get install -y libmagick++-dev

RUN Rscript install_packages.R  
RUN R CMD INSTALL /app/ennet
RUN Rscript -e "devtools::install_github('QinJingLab/GRNIToolS/GRNIToolS')" 


WORKDIR /workspace
CMD ["bash"]