# Creates a Docker image that runs the "lunar" executable

FROM alpine

RUN apk update && apk add --no-cache libstdc++

# Get packages needed to download and build.
RUN apk add --no-cache --virtual .build-deps alpine-sdk

RUN git clone https://github.com/kristopherjohnson/lunar-c.git

WORKDIR /lunar-c

RUN make lunar
RUN cp -f lunar /usr/local/bin/lunar

WORKDIR /

# Remove the build directory and sources.
RUN rm -rf /lunar-c

# Remove the packages we needed.
RUN apk del .build-deps \
  && rm -rf /var/cache/apk/*

CMD lunar

