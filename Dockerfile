# Creates a Docker image that runs the "lunar" executable

FROM alpine as build

# Get packages needed to download and build.
RUN apk add --no-cache --virtual .build-deps alpine-sdk

RUN git clone https://github.com/kristopherjohnson/lunar-c.git

WORKDIR /lunar-c

RUN make lunar

FROM alpine
WORKDIR /
COPY --from=build /lunar-c/lunar /usr/local/bin/lunar
CMD lunar
