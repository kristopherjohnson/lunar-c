# Creates a Docker image that runs the "lunar" executable

FROM alpine AS build

RUN apk add --no-cache gcc musl-dev make

WORKDIR /app
COPY lunar.c Makefile ./

RUN make lunar CFLAGS="-O3 -Wall -static"

FROM scratch
COPY --from=build /app/lunar /lunar
ENTRYPOINT ["/lunar"]
